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
        public Config Config;

        public RNNDecoder(Config config)
        {
            Config = config;
            if (Config.ModelDirection == MODELDIRECTION.BiDirectional)
            {
                Logger.WriteLine("Model Structure: Bi-directional RNN");
                rnn = new BiRNN<Sequence>();
            }
            else
            {
                Logger.WriteLine("Model Structure: Simple RNN");
                rnn = new ForwardRNN<Sequence>();
            }

            rnn.LoadModel(config.ModelFilePath);
            Logger.WriteLine("CRF Model: {0}", rnn.IsCRFTraining);
        }

        public MODELTYPE ModelType => Config.ModelType;

        public int[][] ProcessNBest(Sentence sent, int nbest)
        {
            if (rnn.IsCRFTraining == false)
            {
                throw new ArgumentException("N-best result is only for RNN-CRF model.");
            }

            var seq = Config.BuildSequence(sent);
            var predicted = rnn.DecodeNBestCRF(seq, nbest);

            return predicted;
        }

        public int[] Process(Sentence sent)
        {
            var seq = Config.BuildSequence(sent);
            var predicted = rnn.IsCRFTraining ? rnn.DecodeCRF(seq) : rnn.DecodeNN(seq);

            return predicted;
        }

        public int[] ProcessSeq2Seq(Sentence sent)
        {
            var predicted = rnn.TestSeq2Seq(sent, Config);
            return predicted;
        }

        public float[][] ComputeTopHiddenLayerOutput(Sentence sent)
        {
            var seq = Config.BuildSequence(sent);
            return rnn.ComputeTopHiddenLayerOutput(seq);
        }

        public float[][] ComputeTopHiddenLayerOutput(Sequence seq)
        {
            return rnn.ComputeTopHiddenLayerOutput(seq);
        }

        public int GetTopHiddenLayerSize()
        {
            return rnn.GetTopHiddenLayerSize();
        }
    }
}