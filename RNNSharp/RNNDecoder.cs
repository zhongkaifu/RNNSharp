using System;
using AdvUtils;
using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class RNNDecoder
    {
        RNN<Sequence> rnn;
        public Config Featurizer;
        public MODELTYPE ModelType { get { return Featurizer.ModelType; } }

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

        public int[][] ProcessNBest(Sentence sent, int nbest)
        {
            if (rnn.IsCRFTraining == false)
            {
                throw new ArgumentException("N-best result is only for RNN-CRF model.");
            }

            Sequence seq = Featurizer.ExtractFeatures(sent);
            int[][] predicted = rnn.DecodeNBestCRF(seq, nbest);

            return predicted;
        }


        public int[] Process(Sentence sent)
        {
            Sequence seq = Featurizer.ExtractFeatures(sent);
            int[] predicted;
            if (rnn.IsCRFTraining == true)
            {
                predicted = rnn.DecodeCRF(seq);
            }
            else
            {
                predicted = rnn.DecodeNN(seq);
            }

            return predicted;
        }

        public int[] ProcessSeq2Seq(Sentence sent)
        {
            int[] predicted;
            predicted = rnn.TestSeq2Seq(sent, Featurizer);
            return predicted;
        }

        public List<float[]> ComputeTopHiddenLayerOutput(Sentence sent)
        {
            Sequence seq = Featurizer.ExtractFeatures(sent);
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
