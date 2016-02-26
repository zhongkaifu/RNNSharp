using System;
using System.Threading.Tasks;
using System.IO;
using AdvUtils;
using System.Numerics;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class SimpleRNN : RNN
    {
        protected int bptt;
        protected int bptt_block;
        protected SimpleLayer[] bptt_hidden;
        protected double[][] bptt_fea;
        protected SparseVector[] bptt_inputs = new SparseVector[MAX_RNN_HIST];

        protected Matrix<double> mat_bptt_syn0_w;
        protected Matrix<double> mat_bptt_syn0_ph;
        protected Matrix<double> mat_bptt_synf;

        //Last hidden layer status
        protected SimpleLayer neuLastHidden;
        //Current hidden layer status
        protected SimpleLayer neuHidden;

        //Feature weights
        protected Matrix<double> HiddenBpttWeights { get; set; }
        protected Matrix<double> Input2HiddenWeights { get; set; }
        protected Matrix<double> Feature2HiddenWeights { get; set; }

        //The learning ratio of each weight
        protected Matrix<double> HiddenBpttWeightsLearningRate { get; set; }
        protected Matrix<double> Input2HiddenWeightsLearningRate { get; set; }
        protected Matrix<double> Feature2HiddenWeightsLearningRate { get; set; }

        protected Vector<double> vecMaxGrad;
        protected Vector<double> vecMinGrad;
        protected Vector<double> vecNormalLearningRate;

        public SimpleRNN()
        {
            ModelType = MODELTYPE.SIMPLE;
            GradientCutoff = 15.0;
            Dropout = 0;

            L1 = 30;
            bptt = 5;
            bptt_block = 10;
            bptt_hidden = null;
            bptt_fea = null;

            DenseFeatureSize = 0;

            neuLastHidden = null;
            neuFeatures = null;
            neuHidden = null;
            OutputLayer = null;
        }

        public void setBPTT(int newval) { bptt = newval; }
        public void setBPTTBlock(int newval) { bptt_block = newval; }

        public override void initWeights()
        {
            int b, a;
            for (b = 0; b < L1; b++)
            {
                for (a = 0; a < L0; a++)
                {
                    Input2HiddenWeights[b][a] = RandInitWeight();
                }
            }


            for (b = 0; b < L1; b++)
            {
                for (a = 0; a < DenseFeatureSize; a++)
                {
                    Feature2HiddenWeights[b][a] = RandInitWeight();

                }
            }

            for (b = 0; b < Hidden2OutputWeight.Height; b++)
            {
                for (a = 0; a < L1; a++)
                {
                    Hidden2OutputWeight[b][a] = RandInitWeight();
                }
            }

            for (b = 0; b < L1; b++)
            {
                for (a = 0; a < L1; a++)
                {
                    HiddenBpttWeights[b][a] = RandInitWeight();
                }
            }
        }

        public override SimpleLayer GetHiddenLayer()
        {
            SimpleLayer m = new SimpleLayer(L1);
            for (int i = 0; i < L1; i++)
            {
                m.cellOutput[i] = neuHidden.cellOutput[i];
                m.er[i] = neuHidden.er[i];
                m.mask[i] = neuHidden.mask[i];
            }

            return m;
        }

        private void computeHiddenActivity(bool isTrain)
        {
            Parallel.For(0, L1, parallelOption, a =>
            {
                double cellOutput = neuHidden.cellOutput[a];
                bool mask = neuHidden.mask[a];
                if (mask == true)
                {
                    cellOutput = 0;
                }
                else
                {
                    if (isTrain == false)
                    {
                        cellOutput = cellOutput * (1.0 - Dropout);
                    }

                    if (cellOutput > 50)
                    {
                        cellOutput = 50;  //for numerical stability
                    }
                    else if (cellOutput < -50)
                    {
                        cellOutput = -50;  //for numerical stability
                    }

                    cellOutput = 1.0 / (1.0 + Math.Exp(-cellOutput));
                }
                neuHidden.cellOutput[a] = cellOutput;
            });
        }

        // forward process. output layer consists of tag value
        public override void computeHiddenLayer(State state, bool isTrain = true)
        {
            //keep last hidden layer and erase activations
            neuLastHidden = neuHidden;

            //hidden(t-1) -> hidden(t)
            neuHidden = new SimpleLayer(L1);
            matrixXvectorADD(neuHidden, neuLastHidden, HiddenBpttWeights, L1, L1, 0);

            //Apply feature values on hidden layer
            var sparse = state.SparseData;
            int n = sparse.Count;
            Parallel.For(0, L1, parallelOption, b =>
            {
                //Sparse features:
                //inputs(t) -> hidden(t)
                //Get sparse feature and apply it into hidden layer

                double[] vector_b = Input2HiddenWeights[b];
                double cellOutput = 0;
                for (int i = 0; i < n; i++)
                {
                    var entry = sparse.GetEntry(i);
                    cellOutput += entry.Value * vector_b[entry.Key];
                }
                

                //Dense features:
                //fea(t) -> hidden(t) 
                if (DenseFeatureSize > 0)
                {
                    vector_b = Feature2HiddenWeights[b];
                    for (int j = 0; j < DenseFeatureSize; j++)
                    {
                        cellOutput += neuFeatures[j] * vector_b[j];
                    }
                }

                neuHidden.cellOutput[b] += cellOutput;
            });

            //activate 1      --sigmoid
            computeHiddenActivity(isTrain);
        }

        public override void computeOutput(double[] doutput)
        {
            //Calculate output layer
            matrixXvectorADD(OutputLayer, neuHidden, Hidden2OutputWeight, L2, L1, 0);
            if (doutput != null)
            {
                for (int i = 0; i < L2; i++)
                {
                    doutput[i] = OutputLayer.cellOutput[i];
                }
            }

            //activation 2   --softmax on words
            SoftmaxLayer(OutputLayer);
        }

        public override void ComputeHiddenLayerErr()
        {
            //error output->hidden for words from specific class    	
            matrixXvectorADD(neuHidden, OutputLayer, Hidden2OutputWeight, L1, L2, 1);

            if (Dropout > 0)
            {
                //Apply drop out on error in hidden layer
                for (int i = 0; i < L1; i++)
                {
                    if (neuHidden.mask[i] == true)
                    {
                        neuHidden.er[i] = 0;
                    }
                }
            }
        }

        public override void LearnOutputWeight()
        {
            //Update hidden-output weights
            Parallel.For(0, L2, parallelOption, c =>
            {
                double er = OutputLayer.er[c];
                double[] vector_c = Hidden2OutputWeight[c];
                for (int a = 0; a < L1; a++)
                {
                    double delta = NormalizeGradient(er * neuHidden.cellOutput[a]);
                    double newLearningRate = UpdateLearningRate(Hidden2OutputWeightLearningRate, c, a, delta);

                    vector_c[a] += newLearningRate * delta;
                }
            });
        }

        private void learnBptt(State state)
        {
            for (int step = 0; step < bptt + bptt_block - 2; step++)
            {
                if (null == bptt_inputs[step])
                    break;

                var sparse = bptt_inputs[step];
                var bptt_fea_step = bptt_fea[step];
                var last_bptt_hidden = bptt_hidden[step + 1];
                var last_last_bptt_hidden = bptt_hidden[step + 2];
                Parallel.For(0, L1, parallelOption, a =>
                {
                    // compute hidden layer gradient
                    neuHidden.er[a] *= neuHidden.cellOutput[a] * (1 - neuHidden.cellOutput[a]);

                    //dense weight update fea->0
                    double[] vector_a = null;
                    double er = neuHidden.er[a];
                    Vector<double> vecErr = new Vector<double>(er);

                    int i = 0;
                    if (DenseFeatureSize > 0)
                    {
                        vector_a = mat_bptt_synf[a];
                        i = 0;
                        while (i < DenseFeatureSize - Vector<double>.Count)
                        {
                            Vector<double> v1 = new Vector<double>(bptt_fea_step, i);
                            Vector<double> v2 = new Vector<double>(vector_a, i);
                            v2 += vecErr * v1;
                            v2.CopyTo(vector_a, i);

                            i += Vector<double>.Count;
                        }

                        while (i < DenseFeatureSize)
                        {
                            vector_a[i] += er * bptt_fea_step[i];
                            i++;
                        }
                    }

                    //sparse weight update hidden->input
                    vector_a = mat_bptt_syn0_w[a];
                    for (i = 0; i < sparse.Count; i++)
                    {
                        var entry = sparse.GetEntry(i);
                        vector_a[entry.Key] += er * entry.Value;
                    }

                    //bptt weight update
                    vector_a = mat_bptt_syn0_ph[a];
                    i = 0;
                    while (i < L1 - Vector<double>.Count)
                    {
                        Vector<double> v1 = new Vector<double>(neuLastHidden.cellOutput, i);
                        Vector<double> v2 = new Vector<double>(vector_a, i);
                        v2 += vecErr * v1;
                        v2.CopyTo(vector_a, i);

                        i += Vector<double>.Count;
                    }
					
                    while(i < L1)
                    {
                        vector_a[i] += er * neuLastHidden.cellOutput[i];
                        i++;
                    }

                });

                //propagates errors hidden->input to the recurrent part
                matrixXvectorADD(neuLastHidden, neuHidden, HiddenBpttWeights, L1, L1, 1);

                for (int a = 0; a < L1; a++)
                {
                    //propagate error from time T-n to T-n-1
                    neuHidden.er[a] = neuLastHidden.er[a] + last_bptt_hidden.er[a];
                }
                if (step < bptt + bptt_block - 3)
                {
                    for (int a = 0; a < L1; a++)
                    {
                        neuHidden.cellOutput[a] = last_bptt_hidden.cellOutput[a];
                        neuLastHidden.cellOutput[a] = last_last_bptt_hidden.cellOutput[a];
                    }
                }
            }

            //restore hidden layer after bptt
            bptt_hidden[0].cellOutput.CopyTo(neuHidden.cellOutput, 0);

            Parallel.For(0, L1, parallelOption, b =>
            {
                double[] vector_b = null;
                double[] vector_bf = null;
                double[] vector_lr = null;

                //Update bptt feature weights
                vector_b = HiddenBpttWeights[b];
                vector_bf = mat_bptt_syn0_ph[b];
                vector_lr = HiddenBpttWeightsLearningRate[b];

                int i = 0;
                while (i < L1 - Vector<double>.Count)
                {
                    Vector<double> vecDelta = new Vector<double>(vector_bf, i);
                    Vector<double> vecLearningRate = new Vector<double>(vector_lr, i);
                    Vector<double> vecB = new Vector<double>(vector_b, i);
                    vecDelta = Vector.Min<double>(vecDelta, vecMaxGrad);
                    vecDelta = Vector.Max<double>(vecDelta, vecMinGrad);

                    vecLearningRate += (vecDelta * vecDelta);
                    vecLearningRate.CopyTo(vector_lr, i);
                    vecLearningRate = vecNormalLearningRate / (Vector<double>.One + Vector.SquareRoot<double>(vecLearningRate));

                    vecB += (vecLearningRate * vecDelta);
                    vecB.CopyTo(vector_b, i);

                    Vector<double>.Zero.CopyTo(vector_bf, i);

                    i += Vector<double>.Count;
                }

                while (i < L1)
                {
                    double delta = NormalizeGradient(vector_bf[i]);
                    double newLearningRate = UpdateLearningRate(HiddenBpttWeightsLearningRate, b, i, delta);

                    vector_b[i] += newLearningRate * delta;
                    //Clean bptt weight error
                    vector_bf[i] = 0;

                    i++;
                }

                //Update dense feature weights
                if (DenseFeatureSize > 0)
                {
                    vector_b = Feature2HiddenWeights[b];
                    vector_bf = mat_bptt_synf[b];
                    vector_lr = Feature2HiddenWeightsLearningRate[b];

                    i = 0;
                    while (i < DenseFeatureSize - Vector<double>.Count)
                    {
                        Vector<double> vecDelta = new Vector<double>(vector_bf, i);
                        Vector<double> vecLearningRate = new Vector<double>(vector_lr, i);
                        Vector<double> vecB = new Vector<double>(vector_b, i);
                        vecDelta = Vector.Min<double>(vecDelta, vecMaxGrad);
                        vecDelta = Vector.Max<double>(vecDelta, vecMinGrad);

                        vecLearningRate += (vecDelta * vecDelta);
                        vecLearningRate.CopyTo(vector_lr, i);
                        vecLearningRate = vecNormalLearningRate / (Vector<double>.One + Vector.SquareRoot<double>(vecLearningRate));

                        vecB += (vecLearningRate * vecDelta);
                        vecB.CopyTo(vector_b, i);

                        vecDelta = Vector<double>.Zero;
                        vecDelta.CopyTo(vector_bf, i);

                        i += Vector<double>.Count;
                    }

                    while (i < DenseFeatureSize)
                    {
                        double delta = NormalizeGradient(vector_bf[i]);
                        double newLearningRate = UpdateLearningRate(Feature2HiddenWeightsLearningRate, b, i, delta);

                        vector_b[i] += newLearningRate * delta;
                        //Clean dense feature weights error
                        vector_bf[i] = 0;

                        i++;
                    }
                }

                //Update sparse feature weights
                vector_b = Input2HiddenWeights[b];
                vector_bf = mat_bptt_syn0_w[b];
                for (int step = 0; step < bptt + bptt_block - 2; step++)
                {
                    var sparse = bptt_inputs[step];
                    if (sparse == null)
                        break;

                    for (i = 0; i < sparse.Count; i++)
                    {
                        int pos = sparse.GetEntry(i).Key;

                        double delta = NormalizeGradient(vector_bf[pos]);
                        double newLearningRate = UpdateLearningRate(Input2HiddenWeightsLearningRate, b, pos, delta);

                        vector_b[pos] += newLearningRate * delta;

                        //Clean sparse feature weight error
                        vector_bf[pos] = 0;
                    }
                }
            });
        }


        public void resetBpttMem()
        {
            bptt_inputs = new SparseVector[MAX_RNN_HIST];

            bptt_hidden = new SimpleLayer[bptt + bptt_block + 1];
            for (int i = 0; i < bptt + bptt_block + 1; i++)
            {
                bptt_hidden[i] = new SimpleLayer(L1);
            }

            bptt_fea = new double[bptt + bptt_block + 2][];
            for (int i = 0; i < bptt + bptt_block + 2; i++)
            {
                bptt_fea[i] = new double[DenseFeatureSize];
            }

            mat_bptt_syn0_w = new Matrix<double>(L1, L0);
            mat_bptt_syn0_ph = new Matrix<double>(L1, L1);
            mat_bptt_synf = new Matrix<double>(L1, DenseFeatureSize);
        }

        public override void CleanStatus()
        {
            Hidden2OutputWeightLearningRate = new Matrix<double>(L2, L1);
            Input2HiddenWeightsLearningRate = new Matrix<double>(L1, L0);
            Feature2HiddenWeightsLearningRate = new Matrix<double>(L1, DenseFeatureSize);
            HiddenBpttWeightsLearningRate = new Matrix<double>(L1, L1);

            vecMaxGrad = new Vector<double>(GradientCutoff);
            vecMinGrad = new Vector<double>(-GradientCutoff);
            vecNormalLearningRate = new Vector<double>(LearningRate);
        }
        public override void InitMem()
        {
            CreateCells();

            Hidden2OutputWeight = new Matrix<double>(L2, L1);
            Input2HiddenWeights = new Matrix<double>(L1, L0);
            Feature2HiddenWeights = new Matrix<double>(L1, DenseFeatureSize);
            HiddenBpttWeights = new Matrix<double>(L1, L1);

            Logger.WriteLine("Initializing weights, random value is {0}", rand.NextDouble());// yy debug
            initWeights();

            //Initialize BPTT
            resetBpttMem();
        }

        public override void netReset(bool updateNet = false)   //cleans hidden layer activation + bptt history
        {
            for (int a = 0; a < L1; a++)
            {
                neuHidden.cellOutput[a] = 0.1;
                neuHidden.mask[a] = false;
            }

            if (updateNet == true)
            {
                //Train mode
                SimpleLayer last_bptt_hidden = bptt_hidden[0];
                if (Dropout > 0)
                {
                    for (int a = 0; a < L1; a++)
                    {
                        if (rand.NextDouble() < Dropout)
                        {
                            neuHidden.mask[a] = true;
                        }
                        last_bptt_hidden.cellOutput[a] = neuHidden.cellOutput[a];
                        last_bptt_hidden.er[a] = 0;
                    }
                }
                else
                {
                    for (int a = 0; a < L1; a++)
                    {
                        last_bptt_hidden.cellOutput[a] = neuHidden.cellOutput[a];
                        last_bptt_hidden.er[a] = 0;
                    }
                }

                Array.Clear(bptt_inputs, 0, MAX_RNN_HIST);
            }
        }


        public override void LearnNet(State state, int numStates, int curState)
        {
            int maxBptt = 0;
            for (maxBptt = 0; maxBptt < bptt + bptt_block - 1; maxBptt++)
            {
                if (bptt_inputs[maxBptt] == null)
                {
                    break;
                }
            }

            //Shift memory needed for bptt to next time step, 
            //and save current hidden and feature layer nodes values for bptt
            SimpleLayer last_bptt_hidden = bptt_hidden[maxBptt];
            double[] last_bptt_fea = bptt_fea[maxBptt];
            for (int a = maxBptt; a > 0; a--)
            {
                bptt_inputs[a] = bptt_inputs[a - 1];
                bptt_hidden[a] = bptt_hidden[a - 1];
                bptt_fea[a] = bptt_fea[a - 1];
            }

            bptt_inputs[0] = state.SparseData;
            bptt_hidden[0] = last_bptt_hidden;
            bptt_fea[0] = last_bptt_fea;
            for (int i = 0; i < L1; i++)
            {
                last_bptt_hidden.cellOutput[i] = neuHidden.cellOutput[i];
                last_bptt_hidden.er[i] = neuHidden.er[i];
                last_bptt_hidden.mask[i] = neuHidden.mask[i];
            }

            for (int i = 0; i < DenseFeatureSize; i++)
            {
                last_bptt_fea[i] = neuFeatures[i];
            }

            // time to learn bptt
            if (curState > 0 && ((curState % bptt_block) == 0 || curState == (numStates - 1)))
            {
                learnBptt(state);
            }
        }

        public override void LoadModel(string filename)
        {
            Logger.WriteLine("Loading SimpleRNN model: {0}", filename);

            StreamReader sr = new StreamReader(filename);
            BinaryReader br = new BinaryReader(sr.BaseStream);

            ModelType = (MODELTYPE)br.ReadInt32();
            if (ModelType != MODELTYPE.SIMPLE)
            {
                throw new Exception("Invalidated model format: must be simple RNN");
            }

            ModelDirection = (MODELDIRECTION)br.ReadInt32();

            int iflag = br.ReadInt32();
            if (iflag == 1)
            {
                IsCRFTraining = true;
            }
            else
            {
                IsCRFTraining = false;
            }

            //Load basic parameters
            L0 = br.ReadInt32();
            L1 = br.ReadInt32();
            L2 = br.ReadInt32();
            DenseFeatureSize = br.ReadInt32();

            //Create cells of each layer
            CreateCells();

            //Load weight matrix between each two layer pairs
            Logger.WriteLine("Loading input2hidden weights...");
            Input2HiddenWeights = LoadMatrix(br);

            Logger.WriteLine("Loading bptt hidden weights...");
            HiddenBpttWeights = LoadMatrix(br);

            if (DenseFeatureSize > 0)
            {
                Logger.WriteLine("Loading feature2hidden weights...");
                Feature2HiddenWeights = LoadMatrix(br);
            }

            Logger.WriteLine("Loading hidden2output weights...");
            Hidden2OutputWeight = LoadMatrix(br);

            if (iflag == 1)
            {
                Logger.WriteLine("Loading CRF tag trans weights...");
                CRFTagTransWeights = LoadMatrix(br);
            }

            sr.Close();
        }

        private void CreateCells()
        {
            neuFeatures = new SingleVector(DenseFeatureSize);
            OutputLayer = new SimpleLayer(L2);
            neuHidden = new SimpleLayer(L1);
        }

        // save model as binary format
        public override void SaveModel(string filename) 
        {
            StreamWriter sw = new StreamWriter(filename);
            BinaryWriter fo = new BinaryWriter(sw.BaseStream);

            fo.Write((int)ModelType);
            fo.Write((int)ModelDirection);

            // Signiture , 0 is for RNN or 1 is for RNN-CRF
            int iflag = 0;
            if (IsCRFTraining == true)
            {
                iflag = 1;
            }
            fo.Write(iflag);

            fo.Write(L0);
            fo.Write(L1);
            fo.Write(L2);
            fo.Write(DenseFeatureSize);


            //weight input->hidden
            Logger.WriteLine("Saving input2hidden weights...");
            SaveMatrix(Input2HiddenWeights, fo);

            Logger.WriteLine("Saving bptt hidden weights...");
            SaveMatrix(HiddenBpttWeights, fo);

            if (DenseFeatureSize > 0)
            {
                //weight fea->hidden
                Logger.WriteLine("Saving feature2hidden weights...");
                SaveMatrix(Feature2HiddenWeights, fo);
            }

            //weight hidden->output
            Logger.WriteLine("Saving hidden2output weights...");
            SaveMatrix(Hidden2OutputWeight, fo);

            if (iflag == 1)
            {
                // Save Bigram
                SaveMatrix(CRFTagTransWeights, fo);
            }

            fo.Close();
        }
    }
}
