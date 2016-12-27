using AdvUtils;
using System;
using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class RNNDecoder
    {
        private readonly RNN<Sequence> rnn;
        public Config Featurizer;

        public RNNDecoder(Config featurizer)
        {
            Featurizer = featurizer;
            if (Featurizer.ModelDirection == MODELDIRECTION.BiDirectional)
            {
                Logger.WriteLine("Model Structure: Bi-directional RNN");
                rnn = new BiRNN<Sequence>();
            }
            else
            {
                Logger.WriteLine("Model Structure: Simple RNN");
                rnn = new ForwardRNN<Sequence>();
            }

            rnn.LoadModel(featurizer.ModelFilePath);
            Logger.WriteLine("CRF Model: {0}", rnn.IsCRFTraining);
        }

        public MODELTYPE ModelType => Featurizer.ModelType;

        public int[][] ProcessNBest(Sentence sent, int nbest)
        {
            if (rnn.IsCRFTraining == false)
            {
                throw new ArgumentException("N-best result is only for RNN-CRF model.");
            }

            var seq = Featurizer.ExtractFeatures(sent);
            var predicted = rnn.DecodeNBestCRF(seq, nbest);

            return predicted;
        }

        public int[] Process(Sentence sent)
        {
            var seq = Featurizer.ExtractFeatures(sent);
            var predicted = rnn.IsCRFTraining ? rnn.DecodeCRF(seq) : rnn.DecodeNN(seq);

            return predicted;
        }

        public int[] ProcessSeq2Seq(Sentence sent)
        {
            var predicted = rnn.TestSeq2Seq(sent, Featurizer);
            return predicted;
        }

        public List<float[]> ComputeTopHiddenLayerOutput(Sentence sent)
        {
            var seq = Featurizer.ExtractFeatures(sent);
            return rnn.ComputeTopHiddenLayerOutput(seq);
        }

        public List<float[]> ComputeTopHiddenLayerOutput(Sequence seq)
        {
            return rnn.ComputeTopHiddenLayerOutput(seq);
        }

        public int GetTopHiddenLayerSize()
        {
            return rnn.GetTopHiddenLayerSize();
        }
    }
}