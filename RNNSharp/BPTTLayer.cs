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
        protected float[][] bptt_fea;
        protected SparseVector[] bptt_inputs = new SparseVector[MAX_RNN_HIST];

        //Feature weights
        protected Matrix<float> BpttWeights { get; set; }
        protected Matrix<float> BpttWeightsDelta { get; set; }
        protected Matrix<float> BpttWeightsLearningRate { get; set; }

        public BPTTLayer(BPTTLayerConfig config) : base(config)
        {
            Logger.WriteLine("Initializing BPTT Layer...");
            Logger.WriteLine("Layer Size: {0}", config.LayerSize);
            bptt = config.Bptt + 1;
            bptt_block = 10;

            Logger.WriteLine("BPTT Size: {0}", bptt);
            Logger.WriteLine("BPTT Block Size: {0}", bptt_block);
        }

        public BPTTLayer()
        {
            LayerConfig = new LayerConfig();
        }

        public override void InitializeWeights(int sparseFeatureSize, int denseFeatureSize)
        {
            DenseFeatureSize = denseFeatureSize;
            SparseFeatureSize = sparseFeatureSize;

            if (SparseFeatureSize > 0)
            {
                SparseWeights = new Matrix<float>(LayerSize, SparseFeatureSize);
                SparseWeightsDelta = new Matrix<float>(LayerSize, SparseFeatureSize);
            }

            if (DenseFeatureSize > 0)
            {
                DenseWeights = new Matrix<float>(LayerSize, DenseFeatureSize);
                DenseWeightsDelta = new Matrix<float>(LayerSize, DenseFeatureSize);
            }

            BpttWeights = new Matrix<float>(LayerSize, LayerSize);
            BpttWeightsDelta = new Matrix<float>(LayerSize, LayerSize);

            Logger.WriteLine("Initializing weights, sparse feature size: {0}, dense feature size: {1}, random value is {2}", 
                SparseFeatureSize, DenseFeatureSize, RNNHelper.rand.NextDouble());
            InitWeights();

            //Initialize BPTT
            ResetBpttMem();

        }

        void ResetBpttMem()
        {
            bptt_inputs = new SparseVector[MAX_RNN_HIST];

            bptt_hidden = new SimpleLayer[bptt + bptt_block + 1];
            for (int i = 0; i < bptt + bptt_block + 1; i++)
            {
                bptt_hidden[i] = new SimpleLayer(LayerConfig);
            }

            bptt_fea = new float[bptt + bptt_block + 2][];  
        }

        void InitWeights()
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

            Logger.WriteLine($"Saving BPTT layer, size = '{LayerSize}', sparse feature size = '{SparseFeatureSize}', dense feature size = '{DenseFeatureSize}'");
            RNNHelper.SaveMatrix(BpttWeights, fo);

            if (SparseFeatureSize > 0)
            {
                Logger.WriteLine("Saving sparse feature weights...");
                RNNHelper.SaveMatrix(SparseWeights, fo);
            }

            if (DenseFeatureSize > 0)
            {
                //weight fea->hidden
                Logger.WriteLine("Saving dense feature weights...");
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
                Logger.WriteLine("Loading sparse feature weights...");
                SparseWeights = RNNHelper.LoadMatrix(br);
            }

            if (DenseFeatureSize > 0)
            {
                Logger.WriteLine("Loading dense feature weights...");
                DenseWeights = RNNHelper.LoadMatrix(br);
            }

        }

        public override void CleanLearningRate()
        {      
            SparseWeightsLearningRate = new Matrix<float>(LayerSize, SparseFeatureSize);
            DenseWeightsLearningRate = new Matrix<float>(LayerSize, DenseFeatureSize);
            BpttWeightsLearningRate = new Matrix<float>(LayerSize, LayerSize);

        }

        // forward process. output layer consists of tag value
        public override void ForwardPass(SparseVector sparseFeature, float[] denseFeature, bool isTrain = true)
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
                    float score = 0;
                    if (SparseFeatureSize > 0)
                    {
                        float[] vector_b = SparseWeights[b];
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
                float score = cellOutput[a];
                if (score > 50)
                {
                    score = 50;  //for numerical stability
                }
                else if (score < -50)
                {
                    score = -50;  //for numerical stability
                }

                score = (float)(1.0 / (1.0 + Math.Exp(-score)));
                cellOutput[a] = score;
            });
        }

        public override void BackwardPass(int numStates, int curState)
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
            bptt_fea[0] = new float[DenseFeatureSize];
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
                    float[] vector_a = null;
                    float er2 = er[a];
                    Vector<float> vecErr = new Vector<float>(er2);

                    int i = 0;
                    if (DenseFeatureSize > 0)
                    {
                        vector_a = DenseWeightsDelta[a];
                        i = 0;
                        while (i < DenseFeatureSize - Vector<float>.Count)
                        {
                            Vector<float> v1 = new Vector<float>(bptt_fea_step, i);
                            Vector<float> v2 = new Vector<float>(vector_a, i);
                            v2 += vecErr * v1;
                            v2.CopyTo(vector_a, i);

                            i += Vector<float>.Count;
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
                    while (i < LayerSize - Vector<float>.Count)
                    {
                        Vector<float> v1 = new Vector<float>(previousCellOutput, i);
                        Vector<float> v2 = new Vector<float>(vector_a, i);
                        v2 += vecErr * v1;
                        v2.CopyTo(vector_a, i);

                        i += Vector<float>.Count;
                    }

                    while (i < LayerSize)
                    {
                        vector_a[i] += er2 * previousCellOutput[i];
                        i++;
                    }

                });

                //propagates errors hidden->input to the recurrent part
                float[] previousHiddenErr = new float[LayerSize];
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
                float[] vector_b = null;
                float[] vector_bf = null;
                float[] vector_lr = null;

                //Update bptt feature weights
                vector_b = BpttWeights[b];
                vector_bf = BpttWeightsDelta[b];
                vector_lr = BpttWeightsLearningRate[b];

                int i = 0;
                while (i < LayerSize - Vector<float>.Count)
                {
                    Vector<float> vecDelta = new Vector<float>(vector_bf, i);
                    Vector<float> vecLearningRateWeights = new Vector<float>(vector_lr, i);
                    Vector<float> vecB = new Vector<float>(vector_b, i);

                    //Normalize delta
                    vecDelta = RNNHelper.NormalizeGradient(vecDelta);

                    //Computing learning rate and update its weights
                    Vector<float> vecLearningRate = RNNHelper.UpdateLearningRate(vecDelta, ref vecLearningRateWeights);
                    vecLearningRateWeights.CopyTo(vector_lr, i);

                    //Update weights
                    vecB += vecLearningRate * vecDelta;
                    vecB.CopyTo(vector_b, i);

                    //Clean weights
                    Vector<float>.Zero.CopyTo(vector_bf, i);

                    i += Vector<float>.Count;
                }

                while (i < LayerSize)
                {
                    float delta = RNNHelper.NormalizeGradient(vector_bf[i]);
                    float newLearningRate = RNNHelper.UpdateLearningRate(BpttWeightsLearningRate, b, i, delta);

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
                    while (i < DenseFeatureSize - Vector<float>.Count)
                    {
                        Vector<float> vecDelta = new Vector<float>(vector_bf, i);
                        Vector<float> vecLearningRateWeights = new Vector<float>(vector_lr, i);
                        Vector<float> vecB = new Vector<float>(vector_b, i);

                        //Normalize delta
                        vecDelta = RNNHelper.NormalizeGradient(vecDelta);

                        //Computing learning rate and update its weights
                        Vector<float> vecLearningRate = RNNHelper.UpdateLearningRate(vecDelta, ref vecLearningRateWeights);
                        vecLearningRateWeights.CopyTo(vector_lr, i);

                        //Update weights
                        vecB += vecLearningRate * vecDelta;
                        vecB.CopyTo(vector_b, i);

                        //Clean weights
                        vecDelta = Vector<float>.Zero;
                        vecDelta.CopyTo(vector_bf, i);

                        i += Vector<float>.Count;
                    }

                    while (i < DenseFeatureSize)
                    {
                        float delta = RNNHelper.NormalizeGradient(vector_bf[i]);
                        float newLearningRate = RNNHelper.UpdateLearningRate(DenseWeightsLearningRate, b, i, delta);

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

                            float delta = RNNHelper.NormalizeGradient(vector_bf[pos]);
                            float newLearningRate = RNNHelper.UpdateLearningRate(SparseWeightsLearningRate, b, pos, delta);

                            vector_b[pos] += newLearningRate * delta;

                            //Clean sparse feature weight error
                            vector_bf[pos] = 0;
                        }
                    }
                }
            });
        }

        public override void Reset(bool updateNet = false)   //cleans hidden layer activation + bptt history
        {
            for (int a = 0; a < LayerSize; a++)
            {
                cellOutput[a] = 0.1f;
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
                bptt_fea = new float[bptt + bptt_block + 2][];
            }
        }
    }
}
