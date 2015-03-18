using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class SparseVector : Vector
    {
        KeyValuePair<int, double>[] m_Data;
        int m_Dimension;
        int m_Size;


        public KeyValuePair<int, double> GetEntry(int pos) { return m_Data[pos]; }

        public override int GetDimension() { return m_Dimension; }
        public int GetNumberOfEntries() { return m_Size; }

        public void ChangeValue(int positionInSparseVector, int dimension, double value)
        {
            m_Data[positionInSparseVector] = new KeyValuePair<int, double>(dimension, value);
        }

        public void SetDimension(int s) { m_Dimension = s; }


        public KeyValuePair<int, double>[] GetIndexValues()
        {
            return m_Data;
        }

        public void SetData(Dictionary<int, double> m)
        {
            m_Size = m.Count;
            m_Data = new KeyValuePair<int, double>[m_Size];

            int count = 0;
            foreach (KeyValuePair<int, double> pair in m)
            {
                m_Data[count] = pair;
                count++;
            }
        }

    }
}
