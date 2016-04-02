using System;
using AdvUtils;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class RNNDecoder
    {
        RNN m_Rnn;
        Featurizer m_Featurizer;

        public RNNDecoder(string strModelFileName, Featurizer featurizer)
        {
            MODELTYPE modelType = MODELTYPE.SIMPLE;
            MODELDIRECTION modelDir = MODELDIRECTION.FORWARD;
            int HiddenLayerSize;

            RNN.CheckModelFileType(strModelFileName, out modelType, out modelDir, out HiddenLayerSize);

            if (modelDir == MODELDIRECTION.BI_DIRECTIONAL)
            {
                Logger.WriteLine("Model Structure: Bi-directional RNN");
                if (modelType == MODELTYPE.SIMPLE)
                {
                    m_Rnn = new BiRNN(new SimpleRNN(new SimpleLayer(HiddenLayerSize)), new SimpleRNN(new SimpleLayer(HiddenLayerSize)));
                }
                else
                {
                    m_Rnn = new BiRNN(new LSTMRNN(new LSTMLayer(HiddenLayerSize)), new LSTMRNN(new LSTMLayer(HiddenLayerSize)));
                }
            }
            else
            {
                if (modelType == MODELTYPE.SIMPLE)
                {
                    Logger.WriteLine("Model Structure: Simple RNN");
                    m_Rnn = new SimpleRNN(new SimpleLayer(HiddenLayerSize));
                }
                else
                {
                    Logger.WriteLine("Model Structure: LSTM-RNN");
                    m_Rnn = new LSTMRNN(new LSTMLayer(HiddenLayerSize));
                }
            }

            m_Rnn.LoadModel(strModelFileName);
            Logger.WriteLine("CRF Model: {0}", m_Rnn.IsCRFTraining);
            m_Featurizer = featurizer;
        }


        public int[][] ProcessNBest(Sentence sent, int nbest)
        {
            if (m_Rnn.IsCRFTraining == false)
            {
                throw new ArgumentException("N-best result is only for RNN-CRF model.");
            }

            Sequence seq = m_Featurizer.ExtractFeatures(sent);
            int[][] predicted = m_Rnn.DecodeNBestCRF(seq, nbest);

            return predicted;
        }


        public int[] Process(Sentence sent)
        {
            Sequence seq = m_Featurizer.ExtractFeatures(sent);
            int[] predicted;
            if (m_Rnn.IsCRFTraining == true)
            {
                predicted = m_Rnn.DecodeCRF(seq);
            }
            else
            {
                predicted = m_Rnn.DecodeNN(seq);
            }

            return predicted;
        }
    }
}
