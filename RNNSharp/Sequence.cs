using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class Sequence
    {
        State[] m_States;
        int m_NumStates;

        public int GetSize() { return m_NumStates; }
        public State Get(int i) { return m_States[i]; }


        public int GetDenseDimension()
        {
            if (0 == m_NumStates) return 0;
            else return m_States[0].GetDenseDimension();
        }

        public int GetSparseDimension()
        {
            if (0 == m_NumStates) return 0;
            else return m_States[0].GetSparseDimension();
        }

        public bool SetLabel(Sentence sent, TagSet tagSet)
        {
            List<string[]> features = sent.GetFeatureSet();
            if (features.Count != m_States.Length)
            {
                return false;
            }

            for (int i = 0; i < features.Count; i++)
            {
                string strTagName = features[i][features[i].Length - 1];
                int tagId = tagSet.GetIndex(strTagName);
                if (tagId < 0)
                {
                    Console.WriteLine("Error: tag {0} is unknown.", strTagName);
                    return false;
                }

                m_States[i].SetLabel(tagId);
            }

            return true;
        }

        public void SetSize(int numStates)
        {
            if (m_NumStates != numStates)
            {
                m_NumStates = numStates;
                m_States = null;
                if (m_NumStates > 0)
                {
                    m_States = new State[m_NumStates];
                    for (int i = 0; i < m_NumStates; i++)
                    {
                        m_States[i] = new State();
                    }
                }
            }
        }

    }
}
