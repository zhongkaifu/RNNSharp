using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class DataSet
    {
        List<Sequence> m_Data;
        int m_tagSize;
        List<List<double>> m_LabelBigramTransition;

        /// <summary>
        /// Split current corpus into two parts according given ratio
        /// </summary>
        /// <param name="ratio"></param>
        /// <param name="ds1"></param>
        /// <param name="ds2"></param>
        public void SplitDataSet(double ratio, out DataSet ds1, out DataSet ds2)
        {
            Random rnd = new Random(DateTime.Now.Millisecond);
            ds1 = new DataSet(m_tagSize);
            ds2 = new DataSet(m_tagSize);

            for (int i = 0; i < m_Data.Count; i++)
            {
                if (rnd.NextDouble() < ratio)
                {
                    ds1.Add(m_Data[i]);
                }
                else
                {
                    ds2.Add(m_Data[i]);
                }
            }

            ds1.BuildLabelBigramTransition();
            ds2.BuildLabelBigramTransition();
        }

        public void Add(Sequence sequence) { m_Data.Add(sequence); }

        public void Shuffle()
        {
            Random rnd = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < m_Data.Count; i++)
            {
                int m = rnd.Next() % m_Data.Count;
                Sequence tmp = m_Data[i];
                m_Data[i] = m_Data[m];
                m_Data[m] = tmp;
            }
        }

        public DataSet(int tagSize)
        {
            m_tagSize = tagSize;
            m_Data = new List<Sequence>();
            m_LabelBigramTransition = new List<List<double>>();
        }

        public int GetSize()
        {
            return m_Data.Count;
        }

        public Sequence Get(int i) { return m_Data[i]; }
        public int GetTagSize() { return m_tagSize; }


        public int GetDenseDimension()
        {
            if (0 == m_Data.Count) return 0;
            return m_Data[0].GetDenseDimension();
        }

        public int GetSparseDimension()
        {
            if (0 == m_Data.Count) return 0;
            return m_Data[0].GetSparseDimension();
        }


        public List<List<double>> GetLabelBigramTransition() { return m_LabelBigramTransition; }


        public void BuildLabelBigramTransition(double smooth = 1.0)
        {
            m_LabelBigramTransition = new List<List<double>>();

            for (int i = 0; i < m_tagSize; i++)
            {
                m_LabelBigramTransition.Add(new List<double>());
            }
            for (int i = 0; i < m_tagSize; i++)
            {
                for (int j = 0; j < m_tagSize; j++)
                {
                    m_LabelBigramTransition[i].Add(smooth);
                }
            }

            for (int i = 0; i < m_Data.Count; i++)
            {
                var sequence = m_Data[i];
                if (sequence.GetSize() <= 1)
                    continue;

                int pLabel = sequence.Get(0).GetLabel();
                for (int j = 1; j < sequence.GetSize(); j++)
                {
                    int label = sequence.Get(j).GetLabel();
                    m_LabelBigramTransition[label][pLabel]++;
                    pLabel = label;
                }
            }

            for (int i = 0; i < m_tagSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < m_tagSize; j++)
                {
                    sum += m_LabelBigramTransition[i][j];
                }

                for (int j = 0; j < m_tagSize; j++)
                {
                    m_LabelBigramTransition[i][j] = Math.Log(m_LabelBigramTransition[i][j] / sum);
                }
            }
        }
    }
}
