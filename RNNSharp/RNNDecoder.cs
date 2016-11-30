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
        public Featurizer Featurizer;
        public MODELTYPE ModelType { get; set; }

        public RNNDecoder(string strModelFileName)
        {
            MODELDIRECTION modelDir = MODELDIRECTION.FORWARD;
            MODELTYPE modelType;

            RNNHelper.CheckModelFileType(strModelFileName, out modelDir, out modelType);
            if (modelDir == MODELDIRECTION.BI_DIRECTIONAL)
            {
                Logger.WriteLine("Model Structure: Bi-directional RNN");
                rnn = new BiRNN<Sequence>();
            }
            else
            {
                Logger.WriteLine("Model Structure: Simple RNN");
                rnn = new ForwardRNN<Sequence>();
            }
            ModelType = modelType;

            rnn.LoadModel(strModelFileName);
            Logger.WriteLine("CRF Model: {0}", rnn.IsCRFTraining);
        }

        public void SetFeaturizer(Featurizer featurizer)
        {
            Featurizer = featurizer;
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

        public List<double[]> ComputeTopHiddenLayerOutput(Sentence sent)
        {
            Sequence seq = Featurizer.ExtractFeatures(sent);
            return rnn.ComputeTopHiddenLayerOutput(seq);
        }

        public List<double[]> ComputeTopHiddenLayerOutput(Sequence seq)
        {
            return rnn.ComputeTopHiddenLayerOutput(seq);
        }

        public int GetTopHiddenLayerSize()
        {
            return rnn.GetTopHiddenLayerSize();
        }
    }
}
