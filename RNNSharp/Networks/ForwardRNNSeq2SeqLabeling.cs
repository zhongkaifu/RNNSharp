using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp.Networks
{
    class ForwardRNNSeq2SeqLabeling<T> : ForwardRNNSeq2Seq<T> where T : ISequence
    {
        public ForwardRNNSeq2SeqLabeling()
            :base()
        {

        }

        public override RNN<T> Clone()
        {
            List<SimpleLayer> hiddenLayers = new List<SimpleLayer>();

            foreach (SimpleLayer layer in HiddenLayerList)
            {
                hiddenLayers.Add(layer.CreateLayerSharedWegiths());
            }

            ForwardRNNSeq2SeqLabeling<T> rnn = new ForwardRNNSeq2SeqLabeling<T>();
            rnn.HiddenLayerList = hiddenLayers;
            rnn.OutputLayer = OutputLayer.CreateLayerSharedWegiths();
            rnn.CRFTagTransWeights = CRFTagTransWeights;
            rnn.MaxSeqLength = MaxSeqLength;
            rnn.crfLocker = crfLocker;

            return rnn;
        }

        public override int[] ProcessSequence(ISentence sentence, Config featurizer, RunningMode runningMode, bool outputRawScore, out Matrix<float> m)
        {
            var sequencePair = featurizer.BuildSequence(sentence as SentencePair);
            return TrainSequencePair(sequencePair, runningMode, outputRawScore, out m);
        }
    }
}
