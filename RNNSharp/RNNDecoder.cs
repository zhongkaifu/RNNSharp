using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class RNNDecoder
    {
        RNN m_Rnn;
        Featurizer m_Featurizer;

        public RNNDecoder(string strModelFileName, Featurizer featurizer)
        {
            MODELTYPE model_type = RNN.CheckModelFileType(strModelFileName);

            if (model_type == MODELTYPE.SIMPLE)
            {
                Console.WriteLine("Model Structure: Simple RNN");
                SimpleRNN sRNN = new SimpleRNN();
                m_Rnn = sRNN;
            }
            else
            {
                Console.WriteLine("Model Structure: LSTM-RNN");
                LSTMRNN lstmRNN = new LSTMRNN();
                m_Rnn = lstmRNN;
            }

            m_Rnn.loadNetBin(strModelFileName);
            Console.WriteLine("CRF Model: {0}", m_Rnn.IsCRFModel());
            m_Featurizer = featurizer;
        }


        public int[][] ProcessNBest(Sentence sent, int nbest)
        {
            if (m_Rnn.IsCRFModel() == false)
            {
                return null;
            }

            Sequence seq = m_Featurizer.ExtractFeatures(sent);
            int[][] predicted = m_Rnn.DecodeNBestCRF(seq, nbest);


            //Remove the beginning and end character from result
            int[][] results = new int[nbest][];

            for (int k = 0; k < nbest; k++)
            {
                results[k] = new int[predicted[k].Length - 2];
                for (int i = 1; i < predicted[k].Length - 1; i++)
                {
                    results[k][i - 1] = predicted[k][i];
                }
            }
            return results;
        }


        public int[] Process(Sentence sent)
        {
            Sequence seq = m_Featurizer.ExtractFeatures(sent);
            int[] predicted;
            if (m_Rnn.IsCRFModel() == true)
            {
                predicted = m_Rnn.DecodeCRF(seq);
            }
            else
            {
                predicted = m_Rnn.DecodeNN(seq);
            }

            //Remove the beginning and end character from result
            int[] results = new int[predicted.Length - 2];
            for (int i = 1; i < predicted.Length - 1; i++)
            {
                results[i - 1] = predicted[i];
            }

            return results;
        }
    }
}
