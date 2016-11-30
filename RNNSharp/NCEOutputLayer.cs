using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    class NCEOutputLayer : SimpleLayer
    {
        public long[] accFreqTable;
        public int[] accTagIdTable;
        public int vocab_size;
        public long accTotalFreq;

        public ModelSetting m_modelSetting;
        private int NegativeSampleSize = 10;


        public void BuildStatisticData<T>(DataSet<T> m_trainSet) where T : ISequence
        {
            long train_words = 0;
            vocab_size = 0;
            Dictionary<int, int> tagId2Freq = new Dictionary<int, int>();
            foreach (ISequence seq in m_trainSet.SequenceList)
            {
                State[] States = null;
                if (seq is Sequence)
                {
                    States = (seq as Sequence).States;
                }
                else
                {
                    States = (seq as SequencePair).tgtSequence.States;
                }

                foreach (State state in States)
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
            int accFactor = 1 + (int)(train_words / int.MaxValue);

            SortedDictionary<int, List<int>> freq2TagIdList = new SortedDictionary<int, List<int>>();
            foreach (KeyValuePair<int, int> pair in tagId2Freq)
            {
                if (freq2TagIdList.ContainsKey(pair.Value) == false)
                {
                    freq2TagIdList.Add(pair.Value, new List<int>());
                }
                freq2TagIdList[pair.Value].Add(pair.Key);
            }

            int i = 0;
            foreach (KeyValuePair<int, List<int>> pair in freq2TagIdList.Reverse())
            {
                foreach (int tagId in pair.Value)
                {
                    accTotalFreq += (pair.Key / accFactor);
                    accFreqTable[i] = accTotalFreq;
                    accTagIdTable[i] = tagId;
                    i++;
                }
            }
        }

        public NCEOutputLayer()
        {

        }

        public NCEOutputLayer(int hiddenLayerSize) : base(hiddenLayerSize)
        {

        }
        public NCEOutputLayer(int hiddenLayerSize, ModelSetting modelSetting) : base(hiddenLayerSize)
        {
            Logger.WriteLine("Initializing NCEOutputLayer...");
            m_modelSetting = modelSetting;
            NegativeSampleSize = m_modelSetting.NCESampleSize;
            Logger.WriteLine("Hidden layer size: {0}", hiddenLayerSize);
            Logger.WriteLine("Negative sampling size: {0}", NegativeSampleSize);

            if (NegativeSampleSize > hiddenLayerSize)
            {
                throw new ArgumentException(String.Format("The size of negative sampling('{0}') cannot be greater than the hidden layer size('{1}').", NegativeSampleSize, hiddenLayerSize));
            }
        }

        int SearchAccTermTable(int freq)
        {
            int mid = vocab_size >> 1;
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

        public HashSet<int> negativeSampleWordList = new HashSet<int>();
        public Random rand = new Random();

        public override void ShallowCopyWeightTo(SimpleLayer destLayer)
        {
            NCEOutputLayer layer = destLayer as NCEOutputLayer;
            layer.accFreqTable = accFreqTable;
            layer.accTagIdTable = accTagIdTable;
            layer.vocab_size = vocab_size;
            layer.accTotalFreq = accTotalFreq;

            base.ShallowCopyWeightTo(layer);
        }

        public override void computeLayer(SparseVector sparseFeature, double[] denseFeature, bool isTrain = true)
        {
            if (isTrain == true)
            {
                negativeSampleWordList.Clear();

                foreach (int labelId in LabelShortList)
                {
                    negativeSampleWordList.Add(labelId);
                }

                for (int i = 0; i < NegativeSampleSize; i++)
                {
                    int randomFreq = rand.Next((int)accTotalFreq);
                    int wordId = SearchAccTermTable(randomFreq);
                    while (negativeSampleWordList.Contains(wordId) == true)
                    {
                        wordId = (wordId + 1) % vocab_size;
                    }
                    negativeSampleWordList.Add(wordId);
                }

                if (DenseFeatureSize > 0)
                {
                    DenseFeature = denseFeature;
                    RNNHelper.matrixXvectorADD(cellOutput, denseFeature, DenseWeights, negativeSampleWordList, DenseFeatureSize, true);
                }

                if (SparseFeatureSize > 0)
                {
                    //Apply sparse features
                    SparseFeature = sparseFeature;
                    Parallel.ForEach(negativeSampleWordList, b =>
                    {
                        double score = 0;
                        double[] vector_b = SparseWeights[b];
                        foreach (KeyValuePair<int, float> pair in SparseFeature)
                        {
                            score += pair.Value * vector_b[pair.Key];
                        }
                        cellOutput[b] += score;
                    });
                }


            }
            else
            {
                base.computeLayer(sparseFeature, denseFeature, isTrain);
            }
        }

        public override int GetBestOutputIndex(bool isTrain)
        {
            if (isTrain == true)
            {
                int imax = 0;
                double dmax = double.MinValue;
                foreach (int k in negativeSampleWordList)
                {
                    if (cellOutput[k] > dmax)
                    {
                        dmax = cellOutput[k];
                        imax = k;
                    }
                }
                return imax;
            }
            else
            {
                return base.GetBestOutputIndex(isTrain);
            }
        }

        public override void Softmax(bool isTrain)
        {
            if (isTrain == true)
            {
                double sum = 0;
                foreach (int c in negativeSampleWordList)
                {
                    double cell = cellOutput[c];
                    if (cell > 50) cell = 50;
                    if (cell < -50) cell = -50;
                    double val = Math.Exp(cell);
                    sum += val;
                    cellOutput[c] = val;
                }

                foreach (int c in negativeSampleWordList)
                {
                    cellOutput[c] /= sum;
                }
            }
            else
            {
                base.Softmax(isTrain);
            }
        }

        public override void LearnFeatureWeights(int numStates, int curState)
        {
            if (DenseFeatureSize > 0)
            {
                //Update hidden-output weights
                Parallel.ForEach(negativeSampleWordList, c =>
                {
                    double er2 = er[c];
                    double[] vector_c = DenseWeights[c];
                    for (int a = 0; a < DenseFeatureSize; a++)
                    {
                        double delta = RNNHelper.NormalizeGradient(er2 * DenseFeature[a]);
                        double newLearningRate = RNNHelper.UpdateLearningRate(DenseWeightsLearningRate, c, a, delta);

                        vector_c[a] += newLearningRate * delta;
                    }
                });
            }


            if (SparseFeatureSize > 0)
            {
                //Update hidden-output weights
                Parallel.ForEach(negativeSampleWordList, c =>
                {
                    double er2 = er[c];
                    double[] vector_c = SparseWeights[c];
                    foreach (KeyValuePair<int, float> pair in SparseFeature)
                    {
                        int pos = pair.Key;
                        double val = pair.Value;
                        double delta = RNNHelper.NormalizeGradient(er2 * val);
                        double newLearningRate = RNNHelper.UpdateLearningRate(SparseWeightsLearningRate, c, pos, delta);
                        vector_c[pos] += newLearningRate * delta;
                    }
                });

            }
        }

        public override void ComputeLayerErr(SimpleLayer nextLayer, double[] destErrLayer, double[] srcErrLayer)
        {
            //error output->hidden for words from specific class    	
            RNNHelper.matrixXvectorADDErr(destErrLayer, srcErrLayer, nextLayer.DenseWeights, negativeSampleWordList, nextLayer.LayerSize);
        }

        public override void ComputeLayerErr(SimpleLayer nextLayer)
        {
            //error output->hidden for words from specific class    	
            RNNHelper.matrixXvectorADDErr(er, nextLayer.er, nextLayer.DenseWeights, negativeSampleWordList, nextLayer.LayerSize);
        }

        public override void ComputeLayerErr(Matrix<double> CRFSeqOutput, State state, int timeat)
        {
            if (CRFSeqOutput != null)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                foreach (int c in negativeSampleWordList)
                {
                    er[c] = -CRFSeqOutput[timeat][c];
                }
                er[state.Label] = 1.0 - CRFSeqOutput[timeat][state.Label];
            }
            else
            {
                //For standard RNN
                foreach (int c in negativeSampleWordList)
                {
                    er[c] = -cellOutput[c];
                }
                er[state.Label] = 1.0 - cellOutput[state.Label];
            }

        }
    }
}
