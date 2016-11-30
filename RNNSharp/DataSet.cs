using System;
using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class DataSet<T> where T : ISequence
    {
        public List<T> SequenceList { get; set; }
        public int TagSize { get; set; }
        public List<List<float>> CRFLabelBigramTransition { get; set; }

        public void Shuffle()
        {
            Random rnd = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < SequenceList.Count; i++)
            {
                int m = rnd.Next() % SequenceList.Count;
                T tmp = SequenceList[i];
                SequenceList[i] = SequenceList[m];
                SequenceList[m] = tmp;
            }
        }

        public DataSet(int tagSize)
        {
            TagSize = tagSize;
            SequenceList = new List<T>();
            CRFLabelBigramTransition = new List<List<float>>();
        }

        public int DenseFeatureSize
        {
            get
            {
                if (0 == SequenceList.Count) return 0;
                return SequenceList[0].DenseFeatureSize;
            }
        }

        public int SparseFeatureSize
        {
            get
            {
                if (0 == SequenceList.Count) return 0;
                return SequenceList[0].SparseFeatureSize;
            }
        }

        public void BuildLabelBigramTransition(float smooth = 1.0f)
        {
            CRFLabelBigramTransition = new List<List<float>>();

            for (int i = 0; i < TagSize; i++)
            {
                CRFLabelBigramTransition.Add(new List<float>());
            }
            for (int i = 0; i < TagSize; i++)
            {
                for (int j = 0; j < TagSize; j++)
                {
                    CRFLabelBigramTransition[i].Add(smooth);
                }
            }

            for (int i = 0; i < SequenceList.Count; i++)
            {
                var sequence = SequenceList[i] as Sequence;
                if (sequence.States.Length <= 1)
                    continue;

                int pLabel = sequence.States[0].Label;
                for (int j = 1; j < sequence.States.Length; j++)
                {
                    int label = sequence.States[j].Label;
                    CRFLabelBigramTransition[label][pLabel]++;
                    pLabel = label;
                }
            }

            for (int i = 0; i < TagSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < TagSize; j++)
                {
                    sum += CRFLabelBigramTransition[i][j];
                }

                for (int j = 0; j < TagSize; j++)
                {
                    CRFLabelBigramTransition[i][j] = (float)Math.Log(CRFLabelBigramTransition[i][j] / sum);
                }
            }
        }
    }
}
