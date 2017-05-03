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
        SampledSoftmaxLayerConfig config;

        public SampledSoftmaxLayer(SampledSoftmaxLayerConfig config) : base(config)
        {
            this.config = config;
            NegativeSampleSize = config.NegativeSampleSize;
            if (NegativeSampleSize > LayerSize)
            {
                throw new ArgumentException(
                    $"The size of negative sampling('{NegativeSampleSize}') cannot be greater than the hidden layer size('{LayerSize}').");
            }
        }

        public override SimpleLayer CreateLayerSharedWegiths()
        {
            SampledSoftmaxLayer layer = new SampledSoftmaxLayer(config);
            ShallowCopyWeightTo(layer);
            layer.InitializeInternalTrainingParameters();

            return layer;
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



                                        foreach(var b in negativeSampleWordList)
                    {
                        float score = 0;
                        var vector_b = SparseWeights[b];
                        foreach (var pair in SparseFeature)
                        {
                            score += pair.Value * vector_b[pair.Key];
                        }
                        Cells[b] += score;
                    }
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
            //Update hidden-output weights

                       foreach (var c in negativeSampleWordList)
            {
                UpdateLayerAt(c);
            }

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