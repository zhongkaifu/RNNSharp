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
        public long[] accFreqTable;
        public int[] accTagIdTable;
        public long accTotalFreq;

        public HashSet<int> negativeSampleWordList = new HashSet<int>();
        public Random rand = new Random();
        public int vocab_size;

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

        public void BuildStatisticData<T>(DataSet<T> m_trainSet) where T : ISequence
        {
            long train_words = 0;
            vocab_size = 0;
            var tagId2Freq = new Dictionary<int, int>();
            foreach (ISequence seq in m_trainSet.SequenceList)
            {
                State[] States;
                if (seq is Sequence)
                {
                    States = (seq as Sequence).States;
                }
                else
                {
                    States = (seq as SequencePair).tgtSequence.States;
                }

                foreach (var state in States)
                {
                    if (tagId2Freq.ContainsKey(state.Label) == false)
                    {
                        tagId2Freq.Add(state.Label, 0);
                    }
                    tagId2Freq[state.Label]++;
                    train_words++;
                }
            }

            vocab_size = tagId2Freq.Keys.Count;
            Logger.WriteLine("Vocabulary size: {0}", vocab_size);
            Logger.WriteLine("Training words in total: {0}", train_words);

            accFreqTable = new long[vocab_size];
            accTagIdTable = new int[vocab_size];
            accTotalFreq = 0;

            //Keep accTotalFreq is less than int.MaxValue
            var accFactor = 1 + (int)(train_words / int.MaxValue);

            var freq2TagIdList = new SortedDictionary<int, List<int>>();
            foreach (var pair in tagId2Freq)
            {
                if (freq2TagIdList.ContainsKey(pair.Value) == false)
                {
                    freq2TagIdList.Add(pair.Value, new List<int>());
                }
                freq2TagIdList[pair.Value].Add(pair.Key);
            }

            var i = 0;
            foreach (var pair in freq2TagIdList.Reverse())
            {
                foreach (var tagId in pair.Value)
                {
                    accTotalFreq += pair.Key / accFactor;
                    accFreqTable[i] = accTotalFreq;
                    accTagIdTable[i] = tagId;
                    i++;
                }
            }
        }

        private int SearchAccTermTable(int freq)
        {
            var mid = vocab_size >> 1;
            int left = 0, right = vocab_size - 1;

            while (true)
            {
                if (accFreqTable[mid] < freq)
                {
                    left = mid + 1;
                }
                else if (accFreqTable[mid] > freq)
                {
                    if (mid == 0)
                    {
                        return accTagIdTable[0];
                    }

                    if (accFreqTable[mid - 1] < freq)
                    {
                        return accTagIdTable[mid];
                    }

                    right = mid - 1;
                }
                else
                {
                    return accTagIdTable[mid];
                }

                mid = (left + right) >> 1;
            }
        }

        public override void ShallowCopyWeightTo(SimpleLayer destLayer)
        {
            var layer = destLayer as NCEOutputLayer;
            layer.accFreqTable = accFreqTable;
            layer.accTagIdTable = accTagIdTable;
            layer.vocab_size = vocab_size;
            layer.accTotalFreq = accTotalFreq;

            base.ShallowCopyWeightTo(layer);
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
                    var randomFreq = rand.Next((int)accTotalFreq);
                    var wordId = SearchAccTermTable(randomFreq);
                    while (negativeSampleWordList.Contains(wordId))
                    {
                        wordId = (wordId + 1) % vocab_size;
                    }
                    negativeSampleWordList.Add(wordId);
                }

                if (DenseFeatureSize > 0)
                {
                    DenseFeature = denseFeature;
                    RNNHelper.matrixXvectorADD(cellOutput, denseFeature, DenseWeights, negativeSampleWordList,
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
                        cellOutput[b] += score;
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
                foreach (var k in negativeSampleWordList.Where(k => cellOutput[k] > dmax))
                {
                    dmax = cellOutput[k];
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
                    var cell = cellOutput[c];
                    if (cell > 50) cell = 50;
                    if (cell < -50) cell = -50;
                    var val = (float)Math.Exp(cell);
                    sum += val;
                    cellOutput[c] = val;
                }

                foreach (var c in negativeSampleWordList)
                {
                    cellOutput[c] /= (float)sum;
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
                    var err = er[c];
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
                    var er2 = er[c];
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
            RNNHelper.matrixXvectorADDErr(er, nextLayer.er, nextLayer.DenseWeights, negativeSampleWordList,
                nextLayer.LayerSize);
        }

        public override void ComputeLayerErr(Matrix<float> CRFSeqOutput, State state, int timeat)
        {
            if (CRFSeqOutput != null)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                foreach (var c in negativeSampleWordList)
                {
                    er[c] = -CRFSeqOutput[timeat][c];
                }
                er[state.Label] = (float)(1.0 - CRFSeqOutput[timeat][state.Label]);
            }
            else
            {
                //For standard RNN
                foreach (var c in negativeSampleWordList)
                {
                    er[c] = -cellOutput[c];
                }
                er[state.Label] = (float)(1.0 - cellOutput[state.Label]);
            }
        }
    }
}