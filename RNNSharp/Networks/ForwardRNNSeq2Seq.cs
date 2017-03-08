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
        public ForwardRNNSeq2Seq()
            :base()
        {

        }

        public override void CreateNetwork(List<LayerConfig> hiddenLayersConfig, LayerConfig outputLayerConfig, DataSet<T> TrainingSet, Config featurizer)
        {
            var srcDenseFeatureSize = featurizer.Seq2SeqAutoEncoder.GetTopHiddenLayerSize() * 2;
            var sparseFeatureSize = TrainingSet.SparseFeatureSize;
            sparseFeatureSize += featurizer.Seq2SeqAutoEncoder.Config.SparseFeatureSize;
            Logger.WriteLine("Sparse Feature Format: [{0}][{1}] = {2}", TrainingSet.SparseFeatureSize, featurizer.Seq2SeqAutoEncoder.Config.SparseFeatureSize, sparseFeatureSize);

            HiddenLayerList = CreateLayers(hiddenLayersConfig);

            for (var i = 0; i < HiddenLayerList.Count; i++)
            {
                SimpleLayer layer = HiddenLayerList[i];
                layer.InitializeWeights(sparseFeatureSize, i == 0 ? (srcDenseFeatureSize + TrainingSet.DenseFeatureSize) : (srcDenseFeatureSize + HiddenLayerList[i - 1].LayerSize));
                layer.SetRunningMode(RunningMode.Training);

                Logger.WriteLine($"Create hidden layer {i}: size = {layer.LayerSize}, sparse feature size = {layer.SparseFeatureSize}, dense feature size = {layer.DenseFeatureSize}");
            }

            outputLayerConfig.LayerSize = TrainingSet.TagSize;
            OutputLayer = CreateOutputLayer(outputLayerConfig, sparseFeatureSize, (srcDenseFeatureSize + HiddenLayerList[HiddenLayerList.Count - 1].LayerSize));
            OutputLayer.SetRunningMode(RunningMode.Training);

            Logger.WriteLine($"Create a Forward recurrent neural sequence-to-sequence network with {HiddenLayerList.Count} hidden layers");
        }

        public override RNN<T> Clone()
        {
            List<SimpleLayer> hiddenLayers = new List<SimpleLayer>();

            foreach (SimpleLayer layer in HiddenLayerList)
            {
                hiddenLayers.Add(layer.CreateLayerSharedWegiths());
            }

            ForwardRNNSeq2Seq<T> rnn = new ForwardRNNSeq2Seq<T>();
            rnn.HiddenLayerList = hiddenLayers;
            rnn.OutputLayer = OutputLayer.CreateLayerSharedWegiths();
            rnn.CRFTagTransWeights = CRFTagTransWeights;
            rnn.MaxSeqLength = MaxSeqLength;

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
        private void ExtractSourceSentenceFeature(RNNDecoder decoder, Sequence srcSequence, int targetSparseFeatureSize)
        {
            //Extract dense features from source sequence
            var srcOutputs = decoder.ComputeTopHiddenLayerOutput(srcSequence);
            int srcSequenceDenseFeatureSize = srcOutputs[0].Length;
            int srcSequenceLength = srcOutputs.Length - 1;

            if (srcHiddenAvgOutput == null)
            {
                srcHiddenAvgOutput = new float[srcSequenceDenseFeatureSize * 2];
            }

            var j = 0;
            float[] srcOutputForward = srcOutputs[0];
            float[] srcOutputBackward = srcOutputs[srcSequenceLength];
            while (j < srcSequenceDenseFeatureSize - Vector<float>.Count)
            {
                var vForward = new Vector<float>(srcOutputForward, j);
                var vBackward = new Vector<float>(srcOutputBackward, j);

                vForward.CopyTo(srcHiddenAvgOutput, j);
                vBackward.CopyTo(srcHiddenAvgOutput, srcSequenceDenseFeatureSize + j);

                j += Vector<float>.Count;
            }

            while (j < srcSequenceDenseFeatureSize)
            {
                srcHiddenAvgOutput[j] = srcOutputForward[j];
                srcHiddenAvgOutput[srcSequenceDenseFeatureSize + j] = srcOutputBackward[j];
                j++;
            }

            //Extract sparse features from source sequence
            if (srcSparseFeatures == null)
            {
                srcSparseFeatures = new Dictionary<int, float>();
            }
            else
            {
                srcSparseFeatures.Clear();
            }

            for (var i = 0; i < srcSequence.States.Length; i++)
            {
                foreach (var kv in srcSequence.States[i].SparseFeature)
                {
                    var srcSparseFeatureIndex = kv.Key + targetSparseFeatureSize;

                    if (srcSparseFeatures.ContainsKey(srcSparseFeatureIndex) == false)
                    {
                        srcSparseFeatures.Add(srcSparseFeatureIndex, kv.Value);
                    }
                    else
                    {
                        srcSparseFeatures[srcSparseFeatureIndex] += kv.Value;
                    }
                }
            }
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

            ExtractSourceSentenceFeature(featurizer.Seq2SeqAutoEncoder, srcSequence, curState.SparseFeature.Length);

            var numLayers = HiddenLayerList.Count;
            var predicted = new List<int> { curState.Label };

            CreateDenseFeatureList();
            for (int i = 0; i < numLayers; i++)
            {
                srcHiddenAvgOutput.CopyTo(denseFeaturesList[i], 0);
            }
            srcHiddenAvgOutput.CopyTo(denseFeaturesList[numLayers], 0);

            var sparseVector = new SparseVector();
            while (true)
            {
                //Build sparse features
                sparseVector.Clean();
                sparseVector.SetLength(curState.SparseFeature.Length + srcSequence.SparseFeatureSize);
                sparseVector.AddKeyValuePairData(curState.SparseFeature);
                sparseVector.AddKeyValuePairData(srcSparseFeatures);

                //Compute first layer
                curState.DenseFeature.CopyTo().CopyTo(denseFeaturesList[0], srcHiddenAvgOutput.Length);
                HiddenLayerList[0].ForwardPass(sparseVector, denseFeaturesList[0]);

                //Compute middle layers
                for (var i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    HiddenLayerList[i - 1].Cells.CopyTo(denseFeaturesList[i], srcHiddenAvgOutput.Length);
                    HiddenLayerList[i].ForwardPass(sparseVector, denseFeaturesList[i]);
                }

                //Compute output layer
                HiddenLayerList[numLayers - 1].Cells.CopyTo(denseFeaturesList[numLayers], srcHiddenAvgOutput.Length);
                OutputLayer.ForwardPass(sparseVector, denseFeaturesList[numLayers]);

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

        List<float[]> denseFeaturesList = null;
        float[] srcHiddenAvgOutput;
        Dictionary<int, float> srcSparseFeatures;
        private void CreateDenseFeatureList()
        {
            if (denseFeaturesList == null)
            {
                denseFeaturesList = new List<float[]>();
                for (int i = 0; i < HiddenLayerList.Count; i++)
                {
                    denseFeaturesList.Add(new float[2048]);
                }

                denseFeaturesList.Add(new float[2048]);
            }
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
            ExtractSourceSentenceFeature(pSequence.autoEncoder, srcSequence, tgtSequence.SparseFeatureSize);

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

            CreateDenseFeatureList();
            for (int i = 0; i < numLayers; i++)
            {
                srcHiddenAvgOutput.CopyTo(denseFeaturesList[i], 0);
            }
            srcHiddenAvgOutput.CopyTo(denseFeaturesList[numLayers], 0);

            var sparseVector = new SparseVector();
            for (var curState = 0; curState < numStates; curState++)
            {
                //Build runtime features
                var state = tgtSequence.States[curState];
                SetRuntimeFeatures(state, curState, numStates, predicted);

                //Build sparse features for all layers
                sparseVector.Clean();
                sparseVector.SetLength(tgtSequence.SparseFeatureSize + srcSequence.SparseFeatureSize);
                sparseVector.AddKeyValuePairData(state.SparseFeature);
                sparseVector.AddKeyValuePairData(srcSparseFeatures);

                //Compute first layer
                state.DenseFeature.CopyTo().CopyTo(denseFeaturesList[0], srcHiddenAvgOutput.Length);
                HiddenLayerList[0].ForwardPass(sparseVector, denseFeaturesList[0]);

                //Compute middle layers
                for (var i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    HiddenLayerList[i - 1].Cells.CopyTo(denseFeaturesList[i], srcHiddenAvgOutput.Length);
                    HiddenLayerList[i].ForwardPass(sparseVector, denseFeaturesList[i]);
                }

                //Compute output layer
                HiddenLayerList[numLayers - 1].Cells.CopyTo(denseFeaturesList[numLayers], srcHiddenAvgOutput.Length);
                OutputLayer.ForwardPass(sparseVector, denseFeaturesList[numLayers]);

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
