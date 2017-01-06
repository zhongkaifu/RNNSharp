using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

namespace RNNSharp
{
    internal class NCEOutputLayer : SimpleLayer
    {
        private readonly int NegativeSampleSize = 10;
        public HashSet<int> negativeSampleWordList = new HashSet<int>();
        public Random rand = new Random();

        public NCEOutputLayer()
        {
            LayerConfig = new LayerConfig();
        }

        public NCEOutputLayer(NCELayerConfig config) : base(config)
        {
            NegativeSampleSize = config.NegativeSampleSize;
            if (NegativeSampleSize > LayerSize)
            {
                throw new ArgumentException(
                    $"The size of negative sampling('{NegativeSampleSize}') cannot be greater than the hidden layer size('{LayerSize}').");
            }
        }

        public override void ForwardPass(SparseVector sparseFeature, float[] denseFeature, bool isTrain = true)
        {
            if (isTrain)
            {
                negativeSampleWordList.Clear();

                foreach (var labelId in LabelShortList)
                {
                    negativeSampleWordList.Add(labelId);
                }

                for (var i = 0; i < NegativeSampleSize; i++)
                {
                    var wordId = rand.Next() % LayerSize;
                    while (negativeSampleWordList.Contains(wordId))
                    {
                        wordId = (wordId + 1) % LayerSize;
                    }
                    negativeSampleWordList.Add(wordId);
                }

                if (DenseFeatureSize > 0)
                {
                    DenseFeature = denseFeature;
                    RNNHelper.matrixXvectorADD(Cell, denseFeature, DenseWeights, negativeSampleWordList,
                        DenseFeatureSize, true);
                }

                if (SparseFeatureSize > 0)
                {
                    //Apply sparse features
                    SparseFeature = sparseFeature;
                    Parallel.ForEach(negativeSampleWordList, b =>
                    {
                        float score = 0;
                        var vector_b = SparseWeights[b];
                        foreach (var pair in SparseFeature)
                        {
                            score += pair.Value * vector_b[pair.Key];
                        }
                        Cell[b] += score;
                    });
                }
            }
            else
            {
                base.ForwardPass(sparseFeature, denseFeature, isTrain);
            }
        }

        public override int GetBestOutputIndex(bool isTrain)
        {
            if (isTrain)
            {
                var imax = 0;
                var dmax = double.MinValue;
                foreach (var k in negativeSampleWordList.Where(k => Cell[k] > dmax))
                {
                    dmax = Cell[k];
                    imax = k;
                }
                return imax;
            }
            return base.GetBestOutputIndex(isTrain);
        }

        public override void Softmax(bool isTrain)
        {
            if (isTrain)
            {
                double sum = 0;
                foreach (var c in negativeSampleWordList)
                {
                    var cell = Cell[c];
                    if (cell > 50) cell = 50;
                    if (cell < -50) cell = -50;
                    var val = (float)Math.Exp(cell);
                    sum += val;
                    Cell[c] = val;
                }

                foreach (var c in negativeSampleWordList)
                {
                    Cell[c] /= (float)sum;
                }
            }
            else
            {
                base.Softmax(isTrain);
            }
        }

        public override void BackwardPass(int numStates, int curState)
        {
            if (DenseFeatureSize > 0)
            {
                //Update hidden-output weights
                Parallel.ForEach(negativeSampleWordList, c =>
                {
                    var err = Err[c];
                    var featureWeightCol = DenseWeights[c];
                    var featureWeightsLearningRateCol = DenseWeightsLearningRate[c];
                    var j = 0;
                    while (j < DenseFeatureSize - Vector<float>.Count)
                    {
                        RNNHelper.UpdateFeatureWeights(DenseFeature, featureWeightCol, featureWeightsLearningRateCol,
                            err, j);
                        j += Vector<float>.Count;
                    }

                    while (j < DenseFeatureSize)
                    {
                        var delta = RNNHelper.NormalizeGradient(err * DenseFeature[j]);
                        var newLearningRate = RNNHelper.UpdateLearningRate(DenseWeightsLearningRate, c, j, delta);
                        featureWeightCol[j] += newLearningRate * delta;
                        j++;
                    }
                });
            }

            if (SparseFeatureSize > 0)
            {
                //Update hidden-output weights
                Parallel.ForEach(negativeSampleWordList, c =>
                {
                    var er2 = Err[c];
                    var vector_c = SparseWeights[c];
                    foreach (var pair in SparseFeature)
                    {
                        var pos = pair.Key;
                        var val = pair.Value;
                        var delta = RNNHelper.NormalizeGradient(er2 * val);
                        var newLearningRate = RNNHelper.UpdateLearningRate(SparseWeightsLearningRate, c, pos, delta);
                        vector_c[pos] += newLearningRate * delta;
                    }
                });
            }
        }

        public override void ComputeLayerErr(SimpleLayer nextLayer, float[] destErrLayer, float[] srcErrLayer)
        {
            //error output->hidden for words from specific class
            RNNHelper.matrixXvectorADDErr(destErrLayer, srcErrLayer, nextLayer.DenseWeights, negativeSampleWordList,
                nextLayer.LayerSize);
        }

        public override void ComputeLayerErr(SimpleLayer nextLayer)
        {
            //error output->hidden for words from specific class
            RNNHelper.matrixXvectorADDErr(Err, nextLayer.Err, nextLayer.DenseWeights, negativeSampleWordList,
                nextLayer.LayerSize);
        }

        public override void ComputeLayerErr(Matrix<float> CRFSeqOutput, State state, int timeat)
        {
            if (CRFSeqOutput != null)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                foreach (var c in negativeSampleWordList)
                {
                    Err[c] = -CRFSeqOutput[timeat][c];
                }
                Err[state.Label] = (float)(1.0 - CRFSeqOutput[timeat][state.Label]);
            }
            else
            {
                //For standard RNN
                foreach (var c in negativeSampleWordList)
                {
                    Err[c] = -Cell[c];
                }
                Err[state.Label] = (float)(1.0 - Cell[state.Label]);
            }
        }
    }
}