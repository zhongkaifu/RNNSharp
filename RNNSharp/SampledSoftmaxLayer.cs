using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

namespace RNNSharp
{
    internal class SampledSoftmaxLayer : SoftmaxLayer
    {
        private readonly int NegativeSampleSize = 10;
        public HashSet<int> negativeSampleWordList = new HashSet<int>();
        public Random rand = new Random();

        public SampledSoftmaxLayer(SampledSoftmaxLayerConfig config) : base(config)
        {
            NegativeSampleSize = config.NegativeSampleSize;
            if (NegativeSampleSize > LayerSize)
            {
                throw new ArgumentException(
                    $"The size of negative sampling('{NegativeSampleSize}') cannot be greater than the hidden layer size('{LayerSize}').");
            }
        }

        public override void ForwardPass(SparseVector sparseFeature, float[] denseFeature)
        {
            if (runningMode == RunningMode.Training)
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
                    RNNHelper.matrixXvectorADD(Cells, denseFeature, DenseWeights, negativeSampleWordList, DenseFeatureSize);
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
                        Cells[b] += score;
                    });
                }

                //Softmax
                double sum = 0;
                foreach (var c in negativeSampleWordList)
                {
                    var cell = Cells[c];
                    if (cell > 50) cell = 50;
                    if (cell < -50) cell = -50;
                    var val = (float)Math.Exp(cell);
                    sum += val;
                    Cells[c] = val;
                }

                foreach (var c in negativeSampleWordList)
                {
                    Cells[c] /= (float)sum;
                }
            }
            else
            {
                base.ForwardPass(sparseFeature, denseFeature);
            }
        }

        public override int GetBestOutputIndex()
        {
            if (runningMode == RunningMode.Training)
            {
                var imax = 0;
                var dmax = double.MinValue;
                foreach (var k in negativeSampleWordList.Where(k => Cells[k] > dmax))
                {
                    dmax = Cells[k];
                    imax = k;
                }
                return imax;
            }
            else
            {
                return base.GetBestOutputIndex();
            }
        } 

        public override void BackwardPass()
        {
            if (DenseFeatureSize > 0)
            {
                //Update hidden-output weights
                Parallel.ForEach(negativeSampleWordList, c =>
                {
                    var err = Errs[c];
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
                    var er2 = Errs[c];
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

        public override void ComputeLayerErr(SimpleLayer nextLayer)
        {
            //error output->hidden for words from specific class
            RNNHelper.matrixXvectorADDErr(Errs, nextLayer.Errs, nextLayer.DenseWeights, negativeSampleWordList,
                nextLayer.LayerSize);
        }

        public override void ComputeLayerErr(Matrix<float> CRFSeqOutput, State state, int timeat)
        {
            if (CRFSeqOutput != null)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                foreach (var c in negativeSampleWordList)
                {
                    Errs[c] = -CRFSeqOutput[timeat][c];
                }
                Errs[state.Label] = (float)(1.0 - CRFSeqOutput[timeat][state.Label]);
            }
            else
            {
                //For standard RNN
                foreach (var c in negativeSampleWordList)
                {
                    Errs[c] = -Cells[c];
                }
                Errs[state.Label] = (float)(1.0 - Cells[state.Label]);
            }
        }
    }
}