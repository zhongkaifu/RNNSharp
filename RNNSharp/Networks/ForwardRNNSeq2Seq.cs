using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp.Networks
{
    class ForwardRNNSeq2Seq<T> : ForwardRNN<T> where T : ISequence
    {
        List<float[]>[] denseFeatureGroupsList;
        List<float[]> denseFeatureGroupsOutputLayer;
        List<SparseVector> sparseFeatureGorups;

        public ForwardRNNSeq2Seq()
            : base()
        {

        }

        public override void CreateNetwork(List<LayerConfig> hiddenLayersConfig, LayerConfig outputLayerConfig, DataSet<T> TrainingSet, Config featurizer)
        {
            var sparseFeatureSize = TrainingSet.SparseFeatureSize;
            sparseFeatureSize += featurizer.Seq2SeqAutoEncoder.Config.SparseFeatureSize;
            sparseFeatureGorups = new List<SparseVector>();

            Logger.WriteLine("Sparse Feature Format: [{0}][{1}] = {2}", TrainingSet.SparseFeatureSize, featurizer.Seq2SeqAutoEncoder.Config.SparseFeatureSize, sparseFeatureSize);

            //Create all hidden layers
            HiddenLayerList = CreateLayers(hiddenLayersConfig);

            var srcDenseFeatureSize = featurizer.Seq2SeqAutoEncoder.GetTopHiddenLayerSize() * 2;
            denseFeatureGroupsList = new List<float[]>[HiddenLayerList.Count];
            for (var i = 0; i < HiddenLayerList.Count; i++)
            {
                SimpleLayer layer = HiddenLayerList[i];
                layer.InitializeWeights(sparseFeatureSize, i == 0 ? (srcDenseFeatureSize + TrainingSet.DenseFeatureSize) : (srcDenseFeatureSize + HiddenLayerList[i - 1].LayerSize));
                layer.SetRunningMode(RunningMode.Training);

                Logger.WriteLine($"Create hidden layer {i}: size = {layer.LayerSize}, sparse feature size = {layer.SparseFeatureSize}, dense feature size = {layer.DenseFeatureSize}");

                denseFeatureGroupsList[i] = new List<float[]>();

            }

            //Create output layer
            outputLayerConfig.LayerSize = TrainingSet.TagSize;
            OutputLayer = CreateOutputLayer(outputLayerConfig, sparseFeatureSize, (srcDenseFeatureSize + HiddenLayerList[HiddenLayerList.Count - 1].LayerSize));
            OutputLayer.SetRunningMode(RunningMode.Training);

            Logger.WriteLine($"Create a Forward recurrent neural sequence-to-sequence network with {HiddenLayerList.Count} hidden layers");

            denseFeatureGroupsOutputLayer = new List<float[]>();
        }

        public override RNN<T> Clone()
        {
            List<SimpleLayer> hiddenLayers = new List<SimpleLayer>();
            List<float[]>[] denseFeatureGroupsList = new List<float[]>[HiddenLayerList.Count];

            int i = 0;
            foreach (SimpleLayer layer in HiddenLayerList)
            {
                hiddenLayers.Add(layer.CreateLayerSharedWegiths());
                denseFeatureGroupsList[i] = new List<float[]>();
                i++;
            }

            List<float[]> denseFeatureGroupsOutputLayer = new List<float[]>();

            ForwardRNNSeq2Seq<T> rnn = new ForwardRNNSeq2Seq<T>();
            rnn.HiddenLayerList = hiddenLayers;
            rnn.OutputLayer = OutputLayer.CreateLayerSharedWegiths();
            rnn.CRFWeights = CRFWeights;
            rnn.denseFeatureGroupsList = denseFeatureGroupsList;
            rnn.denseFeatureGroupsOutputLayer = denseFeatureGroupsOutputLayer;
            rnn.sparseFeatureGorups = new List<SparseVector>();
            rnn.MaxSeqLength = MaxSeqLength;
            rnn.bVQ = bVQ;
            rnn.IsCRFTraining = IsCRFTraining;
            if (rnn.IsCRFTraining)
            {
                rnn.InitializeCRFVariablesForTraining();
            }

            return rnn;
        }

        /// <summary>
        /// Extract features from source sequence
        /// </summary>
        /// <param name="decoder"></param>
        /// <param name="srcSequence"></param>
        /// <param name="targetSparseFeatureSize"></param>
        /// <param name="srcHiddenAvgOutput"></param>
        /// <param name="srcSparseFeatures"></param>
        private void ExtractSourceSentenceFeature(RNNDecoder decoder, Sequence srcSequence, int targetSparseFeatureSize, List<float[]> srcDenseFeatureGroups, SparseVector srcSparseFeatures)
        {
            //Extract dense features from source sequence
            var srcOutputs = decoder.ComputeTopHiddenLayerOutput(srcSequence);
            int srcSequenceLength = srcOutputs.Length - 1;

            srcDenseFeatureGroups.Add(srcOutputs[0]);
            srcDenseFeatureGroups.Add(srcOutputs[srcSequenceLength]);

            //Extract sparse features from source sequence
            Dictionary<int, float> srcSparseFeaturesDict = new Dictionary<int, float>();
            for (var i = 0; i < srcSequence.States.Length; i++)
            {
                foreach (var kv in srcSequence.States[i].SparseFeature)
                {
                    var srcSparseFeatureIndex = kv.Key + targetSparseFeatureSize;

                    if (srcSparseFeaturesDict.ContainsKey(srcSparseFeatureIndex) == false)
                    {
                        srcSparseFeaturesDict.Add(srcSparseFeatureIndex, kv.Value);
                    }
                    else
                    {
                        srcSparseFeaturesDict[srcSparseFeatureIndex] += kv.Value;
                    }
                }
            }

            srcSparseFeatures.SetLength(srcSequence.SparseFeatureSize + targetSparseFeatureSize);
            srcSparseFeatures.AddKeyValuePairData(srcSparseFeaturesDict);
        }

        public override int[] ProcessSequence(ISentence sentence, Config featurizer, RunningMode runningMode, bool outputRawScore, out Matrix<float> m)
        {
            if (runningMode == RunningMode.Training)
            {
                var sequencePair = featurizer.BuildSequence(sentence as SentencePair);
                return TrainSequencePair(sequencePair, runningMode, outputRawScore, out m);
            }
            else
            {
                return PredictTargetSentence(sentence as Sentence, featurizer, out m);
            }
        }

        private int[] PredictTargetSentence(Sentence sentence, Config featurizer, out Matrix<float> m)
        {
            m = null;

            var curState = featurizer.BuildState(new[] { "<s>" });
            curState.Label = featurizer.TagSet.GetIndex("<s>");

            //Reset all layers
            foreach (var layer in HiddenLayerList)
            {
                layer.Reset();
            }

            //Extract features from source sentence
            var srcSequence = featurizer.Seq2SeqAutoEncoder.Config.BuildSequence(sentence);
            List<float[]> srcDenseFeatureGorups = new List<float[]>();
            SparseVector srcSparseFeatures = new SparseVector();
            ExtractSourceSentenceFeature(featurizer.Seq2SeqAutoEncoder, srcSequence, curState.SparseFeature.Length, srcDenseFeatureGorups, srcSparseFeatures);

            var numLayers = HiddenLayerList.Count;
            var predicted = new List<int> { curState.Label };

            //Set sparse feature group from source sequence
            sparseFeatureGorups.Clear();
            sparseFeatureGorups.Add(srcSparseFeatures);
            sparseFeatureGorups.Add(null);
            int targetSparseFeatureIndex = sparseFeatureGorups.Count - 1;

            //Set dense feature groups from source sequence
            for (var i = 0; i < numLayers; i++)
            {
                denseFeatureGroupsList[i].Clear();
                denseFeatureGroupsList[i].AddRange(srcDenseFeatureGorups);
                denseFeatureGroupsList[i].Add(null);
            }
            denseFeatureGroupsOutputLayer.Clear();
            denseFeatureGroupsOutputLayer.AddRange(srcDenseFeatureGorups);
            denseFeatureGroupsOutputLayer.Add(null);
            int targetDenseFeatureIndex = denseFeatureGroupsOutputLayer.Count - 1;

            while (true)
            {
                //Set sparse feature groups
                sparseFeatureGorups[targetSparseFeatureIndex] = curState.SparseFeature;

                //Compute first layer
                denseFeatureGroupsList[0][targetDenseFeatureIndex] = curState.DenseFeature.CopyTo();
                HiddenLayerList[0].ForwardPass(sparseFeatureGorups, denseFeatureGroupsList[0]);

                //Compute middle layers
                for (var i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    denseFeatureGroupsList[i][targetDenseFeatureIndex] = HiddenLayerList[i - 1].Cells;
                    HiddenLayerList[i].ForwardPass(sparseFeatureGorups, denseFeatureGroupsList[i]);
                }

                //Compute output layer
                denseFeatureGroupsOutputLayer[targetDenseFeatureIndex] = HiddenLayerList[numLayers - 1].Cells;
                OutputLayer.ForwardPass(sparseFeatureGorups, denseFeatureGroupsOutputLayer);


                var nextTagId = OutputLayer.GetBestOutputIndex();
                var nextWord = featurizer.TagSet.GetTagName(nextTagId);

                curState = featurizer.BuildState(new[] { nextWord });
                curState.Label = nextTagId;

                predicted.Add(nextTagId);

                if (nextWord == "</s>" || predicted.Count >= 100)
                {
                    break;
                }
            }

            return predicted.ToArray();
        }

        public override int[] ProcessSequence(ISequence sequence, RunningMode runningMode, bool outputRawScore, out Matrix<float> m)
        {
            return TrainSequencePair(sequence, runningMode, outputRawScore, out m);
        }

        private int[] TrainSequencePair(ISequence sequence, RunningMode runningMode, bool outputRawScore, out Matrix<float> m)
        {
            SequencePair pSequence = sequence as SequencePair;
            var tgtSequence = pSequence.tgtSequence;

            //Reset all layers
            foreach (var layer in HiddenLayerList)
            {
                layer.Reset();
            }

            Sequence srcSequence;

            //Extract features from source sentences
            srcSequence = pSequence.autoEncoder.Config.BuildSequence(pSequence.srcSentence);
            List<float[]> srcDenseFeatureGorups = new List<float[]>();
            SparseVector srcSparseFeatures = new SparseVector();
            ExtractSourceSentenceFeature(pSequence.autoEncoder, srcSequence, tgtSequence.SparseFeatureSize, srcDenseFeatureGorups, srcSparseFeatures);

            var numStates = pSequence.tgtSequence.States.Length;
            var numLayers = HiddenLayerList.Count;
            var predicted = new int[numStates];

            m = outputRawScore ? new Matrix<float>(numStates, OutputLayer.LayerSize) : null;

            //Set target sentence labels into short list in output layer
            OutputLayer.LabelShortList.Clear();
            foreach (var state in tgtSequence.States)
            {
                OutputLayer.LabelShortList.Add(state.Label);
            }

            //Set sparse feature group from source sequence
            sparseFeatureGorups.Clear();
            sparseFeatureGorups.Add(srcSparseFeatures);
            sparseFeatureGorups.Add(null);
            int targetSparseFeatureIndex = sparseFeatureGorups.Count - 1;

            //Set dense feature groups from source sequence
            for (var i = 0; i < numLayers; i++)
            {
                denseFeatureGroupsList[i].Clear();
                denseFeatureGroupsList[i].AddRange(srcDenseFeatureGorups);
                denseFeatureGroupsList[i].Add(null);
            }
            denseFeatureGroupsOutputLayer.Clear();
            denseFeatureGroupsOutputLayer.AddRange(srcDenseFeatureGorups);
            denseFeatureGroupsOutputLayer.Add(null);
            int targetDenseFeatureIndex = denseFeatureGroupsOutputLayer.Count - 1;

            for (var curState = 0; curState < numStates; curState++)
            {
                var state = tgtSequence.States[curState];

                //Set sparse feature groups
                sparseFeatureGorups[targetSparseFeatureIndex] = state.SparseFeature;

                //Compute first layer
                denseFeatureGroupsList[0][targetDenseFeatureIndex] = state.DenseFeature.CopyTo();
                HiddenLayerList[0].ForwardPass(sparseFeatureGorups, denseFeatureGroupsList[0]);

                //Compute middle layers
                for (var i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    denseFeatureGroupsList[i][targetDenseFeatureIndex] = HiddenLayerList[i - 1].Cells;
                    HiddenLayerList[i].ForwardPass(sparseFeatureGorups, denseFeatureGroupsList[i]);
                }

                //Compute output layer
                denseFeatureGroupsOutputLayer[targetDenseFeatureIndex] = HiddenLayerList[numLayers - 1].Cells;
                OutputLayer.ForwardPass(sparseFeatureGorups, denseFeatureGroupsOutputLayer);

                if (m != null)
                {
                    OutputLayer.Cells.CopyTo(m[curState], 0);
                }

                predicted[curState] = OutputLayer.GetBestOutputIndex();

                if (runningMode == RunningMode.Training)
                {
                    // error propogation
                    OutputLayer.ComputeLayerErr(CRFSeqOutput, state, curState);

                    //propogate errors to each layer from output layer to input layer
                    HiddenLayerList[numLayers - 1].ComputeLayerErr(OutputLayer);
                    for (var i = numLayers - 2; i >= 0; i--)
                    {
                        HiddenLayerList[i].ComputeLayerErr(HiddenLayerList[i + 1]);
                    }

                    //Update net weights
                    OutputLayer.BackwardPass();
                    for (var i = 0; i < numLayers; i++)
                    {
                        HiddenLayerList[i].BackwardPass();
                    }

                }
            }

            return predicted;
        }
    }
}
