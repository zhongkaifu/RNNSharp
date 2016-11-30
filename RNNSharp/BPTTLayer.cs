using System;
using System.Threading.Tasks;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.IO;
using AdvUtils;
using System.Collections.Generic;
/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class BPTTLayer : SimpleLayer
    {
        protected const int MAX_RNN_HIST = 64;

        public int bptt;
        public int bptt_block;
        protected SimpleLayer[] bptt_hidden;
        protected double[][] bptt_fea;
        protected SparseVector[] bptt_inputs = new SparseVector[MAX_RNN_HIST];

        //Feature weights
        protected Matrix<double> BpttWeights { get; set; }
        protected Matrix<double> BpttWeightsDelta { get; set; }
        protected Matrix<double> BpttWeightsLearningRate { get; set; }

        public BPTTLayer(int hiddenLayerSize, ModelSetting modelsetting) : base(hiddenLayerSize)
        {
            Logger.WriteLine("Initializing BPTT Layer...");
            Logger.WriteLine("Layer Size: {0}", hiddenLayerSize);
            bptt = modelsetting.Bptt + 1;
            bptt_block = 10;

            Logger.WriteLine("BPTT Size: {0}", bptt);
            Logger.WriteLine("BPTT Block Size: {0}", bptt_block);
        }

        public BPTTLayer()
        {

        }

        public override void InitializeWeights(int sparseFeatureSize, int denseFeatureSize)
        {
            DenseFeatureSize = denseFeatureSize;
            SparseFeatureSize = sparseFeatureSize;

            if (SparseFeatureSize > 0)
            {
                SparseWeights = new Matrix<double>(LayerSize, SparseFeatureSize);
                SparseWeightsDelta = new Matrix<double>(LayerSize, SparseFeatureSize);
            }

            if (DenseFeatureSize > 0)
            {
                DenseWeights = new Matrix<double>(LayerSize, DenseFeatureSize);
                DenseWeightsDelta = new Matrix<double>(LayerSize, DenseFeatureSize);
            }

            BpttWeights = new Matrix<double>(LayerSize, LayerSize);
            BpttWeightsDelta = new Matrix<double>(LayerSize, LayerSize);

            Logger.WriteLine("Initializing weights, sparse feature size: {0}, dense feature size: {1}, random value is {2}", 
                SparseFeatureSize, DenseFeatureSize, RNNHelper.rand.NextDouble());
            initWeights();

            //Initialize BPTT
            resetBpttMem();

        }

        public void resetBpttMem()
        {
            bptt_inputs = new SparseVector[MAX_RNN_HIST];

            bptt_hidden = new SimpleLayer[bptt + bptt_block + 1];
            for (int i = 0; i < bptt + bptt_block + 1; i++)
            {
                bptt_hidden[i] = new SimpleLayer(LayerSize);
            }

            bptt_fea = new double[bptt + bptt_block + 2][];  
        }

        public void initWeights()
        {
            int b, a;
            for (b = 0; b < LayerSize; b++)
            {
                for (a = 0; a < SparseFeatureSize; a++)
                {
                    SparseWeights[b][a] = RNNHelper.RandInitWeight();
                }
            }


            for (b = 0; b < LayerSize; b++)
            {
                for (a = 0; a < DenseFeatureSize; a++)
                {
                    DenseWeights[b][a] = RNNHelper.RandInitWeight();

                }
            }

            for (b = 0; b < LayerSize; b++)
            {
                for (a = 0; a < LayerSize; a++)
                {
                    BpttWeights[b][a] = RNNHelper.RandInitWeight();
                }
            }
        }

        public override void Save(BinaryWriter fo)
        {
            fo.Write(LayerSize);
            fo.Write(SparseFeatureSize);
            fo.Write(DenseFeatureSize);

            Logger.WriteLine("Saving bptt hidden weights...");
            RNNHelper.SaveMatrix(BpttWeights, fo);

            if (SparseFeatureSize > 0)
            {
                Logger.WriteLine("Saving input2hidden weights...");
                RNNHelper.SaveMatrix(SparseWeights, fo);
            }

            if (DenseFeatureSize > 0)
            {
                //weight fea->hidden
                Logger.WriteLine("Saving feature2hidden weights...");
                RNNHelper.SaveMatrix(DenseWeights, fo);
            }
        }

        public override void Load(BinaryReader br)
        {
            //Load basic parameters
            LayerSize = br.ReadInt32();
            SparseFeatureSize = br.ReadInt32();
            DenseFeatureSize = br.ReadInt32();

            AllocateMemoryForCells();

            Logger.WriteLine("Loading bptt hidden weights...");
            BpttWeights = RNNHelper.LoadMatrix(br);

            if (SparseFeatureSize > 0)
            {
                Logger.WriteLine("Loading input2hidden weights...");
                SparseWeights = RNNHelper.LoadMatrix(br);
            }

            if (DenseFeatureSize > 0)
            {
                Logger.WriteLine("Loading feature2hidden weights...");
                DenseWeights = RNNHelper.LoadMatrix(br);
            }

        }

        public override void CleanLearningRate()
        {      
            SparseWeightsLearningRate = new Matrix<double>(LayerSize, SparseFeatureSize);
            DenseWeightsLearningRate = new Matrix<double>(LayerSize, DenseFeatureSize);
            BpttWeightsLearningRate = new Matrix<double>(LayerSize, LayerSize);

            RNNHelper.vecMaxGrad = new Vector<double>(RNNHelper.GradientCutoff);
            RNNHelper.vecMinGrad = new Vector<double>(-RNNHelper.GradientCutoff);
            RNNHelper.vecNormalLearningRate = new Vector<double>(RNNHelper.LearningRate);
        }

        // forward process. output layer consists of tag value
        public override void computeLayer(SparseVector sparseFeature, double[] denseFeature, bool isTrain = true)
        {
            //keep last hidden layer and erase activations
            cellOutput.CopyTo(previousCellOutput, 0);

            //Apply previous feature to current time
            //hidden(t-1) -> hidden(t)
            RNNHelper.matrixXvectorADD(cellOutput, previousCellOutput, BpttWeights, LayerSize, LayerSize);

            //Apply features on hidden layer
            SparseFeature = sparseFeature;
            DenseFeature = denseFeature;

            if (SparseFeatureSize > 0)
            {
                //Apply sparse features
                Parallel.For(0, LayerSize, parallelOption, b =>
                {
                    double score = 0;
                    if (SparseFeatureSize > 0)
                    {
                        double[] vector_b = SparseWeights[b];
                        foreach (KeyValuePair<int, float> pair in SparseFeature)
                        {
                            score += pair.Value * vector_b[pair.Key];
                        }
                    }
                    cellOutput[b] += score;
                });
            }

            if (DenseFeatureSize > 0)
            {
                //Apply dense features
                RNNHelper.matrixXvectorADD(cellOutput, DenseFeature, DenseWeights, LayerSize, DenseFeatureSize, false);
            }

            //activate layer
            activityLayer();
        }

        private void activityLayer()
        {
            Parallel.For(0, LayerSize, parallelOption, a =>
            {
                double score = cellOutput[a];
                if (score > 50)
                {
                    score = 50;  //for numerical stability
                }
                else if (score < -50)
                {
                    score = -50;  //for numerical stability
                }

                score = 1.0 / (1.0 + Math.Exp(-score));
                cellOutput[a] = score;
            });
        }

        public override void LearnFeatureWeights(int numStates, int curState)
        {
            int maxBptt = 0;
            for (maxBptt = 0; maxBptt < bptt + bptt_block - 1; maxBptt++)
            {
                if (bptt_inputs[maxBptt] == null && bptt_fea[maxBptt] == null)
                {
                    break;
                }
            }

            //Shift memory needed for bptt to next time step, 
            //and save current hidden and feature layer nodes values for bptt
            SimpleLayer last_bptt_hidden = bptt_hidden[maxBptt];
            for (int a = maxBptt; a > 0; a--)
            {
                bptt_inputs[a] = bptt_inputs[a - 1];
                bptt_hidden[a] = bptt_hidden[a - 1];
                bptt_fea[a] = bptt_fea[a - 1];
            }

            bptt_inputs[0] = SparseFeature;
            bptt_hidden[0] = last_bptt_hidden;
            bptt_fea[0] = new double[DenseFeatureSize];
            for (int i = 0; i < LayerSize; i++)
            {
                last_bptt_hidden.cellOutput[i] = cellOutput[i];
                last_bptt_hidden.er[i] = er[i];
            }

            for (int i = 0; i < DenseFeatureSize; i++)
            {
                bptt_fea[0][i] = DenseFeature[i];
            }

            // time to learn bptt
            if (curState > 0 && ((curState % bptt_block) == 0 || curState == (numStates - 1)))
            {
                learnBptt();
            }
        }

        private void learnBptt()
        {
            for (int step = 0; step < bptt + bptt_block - 2; step++)
            {
                if (null == bptt_inputs[step] && null == bptt_fea[step])
                    break;

                var sparse = bptt_inputs[step];
                var bptt_fea_step = bptt_fea[step];
                var last_bptt_hidden = bptt_hidden[step + 1];
                var last_last_bptt_hidden = bptt_hidden[step + 2];
                Parallel.For(0, LayerSize, parallelOption, a =>
                {
                    // compute hidden layer gradient
                    er[a] *= cellOutput[a] * (1 - cellOutput[a]);

                    //dense weight update fea->0
                    double[] vector_a = null;
                    double er2 = er[a];
                    Vector<double> vecErr = new Vector<double>(er2);

                    int i = 0;
                    if (DenseFeatureSize > 0)
                    {
                        vector_a = DenseWeightsDelta[a];
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
                            vector_a[i] += er2 * bptt_fea_step[i];
                            i++;
                        }
                    }

                    if (SparseFeatureSize > 0)
                    {
                        //sparse weight update hidden->input
                        vector_a = SparseWeightsDelta[a];
                        foreach (KeyValuePair<int, float> pair in sparse)
                        {
                            vector_a[pair.Key] += er2 * pair.Value;
                        }
                    }

                    //bptt weight update
                    vector_a = BpttWeightsDelta[a];
                    i = 0;
                    while (i < LayerSize - Vector<double>.Count)
                    {
                        Vector<double> v1 = new Vector<double>(previousCellOutput, i);
                        Vector<double> v2 = new Vector<double>(vector_a, i);
                        v2 += vecErr * v1;
                        v2.CopyTo(vector_a, i);

                        i += Vector<double>.Count;
                    }

                    while (i < LayerSize)
                    {
                        vector_a[i] += er2 * previousCellOutput[i];
                        i++;
                    }

                });

                //propagates errors hidden->input to the recurrent part
                double[] previousHiddenErr = new double[LayerSize];
                RNNHelper.matrixXvectorADDErr(previousHiddenErr, er, BpttWeights, LayerSize, LayerSize);

                for (int a = 0; a < LayerSize; a++)
                {
                    //propagate error from time T-n to T-n-1
                    er[a] = previousHiddenErr[a] + last_bptt_hidden.er[a];
                }
                if (step < bptt + bptt_block - 3)
                {
                    for (int a = 0; a < LayerSize; a++)
                    {
                        cellOutput[a] = last_bptt_hidden.cellOutput[a];
                        previousCellOutput[a] = last_last_bptt_hidden.cellOutput[a];
                    }
                }
            }

            //restore hidden layer after bptt
            bptt_hidden[0].cellOutput.CopyTo(cellOutput, 0);

            Parallel.For(0, LayerSize, parallelOption, b =>
            {
                double[] vector_b = null;
                double[] vector_bf = null;
                double[] vector_lr = null;

                //Update bptt feature weights
                vector_b = BpttWeights[b];
                vector_bf = BpttWeightsDelta[b];
                vector_lr = BpttWeightsLearningRate[b];

                int i = 0;
                while (i < LayerSize - Vector<double>.Count)
                {
                    Vector<double> vecDelta = new Vector<double>(vector_bf, i);
                    Vector<double> vecLearningRateWeights = new Vector<double>(vector_lr, i);
                    Vector<double> vecB = new Vector<double>(vector_b, i);

                    //Normalize delta
                    vecDelta = RNNHelper.NormalizeGradient(vecDelta);

                    //Computing learning rate and update its weights
                    Vector<double> vecLearningRate = RNNHelper.UpdateLearningRate(vecDelta, ref vecLearningRateWeights);
                    vecLearningRateWeights.CopyTo(vector_lr, i);

                    //Update weights
                    vecB += vecLearningRate * vecDelta;
                    vecB.CopyTo(vector_b, i);

                    //Clean weights
                    Vector<double>.Zero.CopyTo(vector_bf, i);

                    i += Vector<double>.Count;
                }

                while (i < LayerSize)
                {
                    double delta = RNNHelper.NormalizeGradient(vector_bf[i]);
                    double newLearningRate = RNNHelper.UpdateLearningRate(BpttWeightsLearningRate, b, i, delta);

                    vector_b[i] += newLearningRate * delta;
                    //Clean bptt weight error
                    vector_bf[i] = 0;

                    i++;
                }

                //Update dense feature weights
                if (DenseFeatureSize > 0)
                {
                    vector_b = DenseWeights[b];
                    vector_bf = DenseWeightsDelta[b];
                    vector_lr = DenseWeightsLearningRate[b];

                    i = 0;
                    while (i < DenseFeatureSize - Vector<double>.Count)
                    {
                        Vector<double> vecDelta = new Vector<double>(vector_bf, i);
                        Vector<double> vecLearningRateWeights = new Vector<double>(vector_lr, i);
                        Vector<double> vecB = new Vector<double>(vector_b, i);

                        //Normalize delta
                        vecDelta = RNNHelper.NormalizeGradient(vecDelta);

                        //Computing learning rate and update its weights
                        Vector<double> vecLearningRate = RNNHelper.UpdateLearningRate(vecDelta, ref vecLearningRateWeights);
                        vecLearningRateWeights.CopyTo(vector_lr, i);

                        //Update weights
                        vecB += vecLearningRate * vecDelta;
                        vecB.CopyTo(vector_b, i);

                        //Clean weights
                        vecDelta = Vector<double>.Zero;
                        vecDelta.CopyTo(vector_bf, i);

                        i += Vector<double>.Count;
                    }

                    while (i < DenseFeatureSize)
                    {
                        double delta = RNNHelper.NormalizeGradient(vector_bf[i]);
                        double newLearningRate = RNNHelper.UpdateLearningRate(DenseWeightsLearningRate, b, i, delta);

                        vector_b[i] += newLearningRate * delta;
                        //Clean dense feature weights error
                        vector_bf[i] = 0;

                        i++;
                    }
                }

                if (SparseFeatureSize > 0)
                {
                    //Update sparse feature weights
                    vector_b = SparseWeights[b];
                    vector_bf = SparseWeightsDelta[b];
                    for (int step = 0; step < bptt + bptt_block - 2; step++)
                    {
                        var sparse = bptt_inputs[step];
                        if (sparse == null)
                            break;

                        foreach (KeyValuePair<int, float> pair in sparse)
                        {
                            int pos = pair.Key;

                            double delta = RNNHelper.NormalizeGradient(vector_bf[pos]);
                            double newLearningRate = RNNHelper.UpdateLearningRate(SparseWeightsLearningRate, b, pos, delta);

                            vector_b[pos] += newLearningRate * delta;

                            //Clean sparse feature weight error
                            vector_bf[pos] = 0;
                        }
                    }
                }
            });
        }

        public override void netReset(bool updateNet = false)   //cleans hidden layer activation + bptt history
        {
            for (int a = 0; a < LayerSize; a++)
            {
                cellOutput[a] = 0.1;
            }

            if (updateNet == true)
            {
                //Train mode
                SimpleLayer last_bptt_hidden = bptt_hidden[0];
                for (int a = 0; a < LayerSize; a++)
                {
                    last_bptt_hidden.cellOutput[a] = cellOutput[a];
                    last_bptt_hidden.er[a] = 0;
                }

                Array.Clear(bptt_inputs, 0, MAX_RNN_HIST);
                bptt_fea = new double[bptt + bptt_block + 2][];
            }
        }
    }
}
