using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class SparseVector : SingleVector
    {
        KeyValuePair<int, float>[] m_Data;
        int m_Dimension;
        int m_Size;

        public KeyValuePair<int, float> GetEntry(int pos) { return m_Data[pos]; }

        public override int GetDimension() { return m_Dimension; }
        public int GetNumberOfEntries() { return m_Size; }

        public void ChangeValue(int positionInSparseVector, int dimension, float value)
        {
            m_Data[positionInSparseVector] = new KeyValuePair<int, float>(dimension, value);
        }

        public void SetDimension(int s) { m_Dimension = s; }

        public void SetData(Dictionary<int, float> m)
        {
            m_Size = m.Count;
            m_Data = new KeyValuePair<int, float>[m_Size];

            int count = 0;
            foreach (KeyValuePair<int, float> pair in m)
            {
                m_Data[count] = pair;
                count++;
            }
        }

    }
}
