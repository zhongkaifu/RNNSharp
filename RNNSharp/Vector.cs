using System;
using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class VectorBase
    {
        public virtual int GetDimension()
        {
            return 0;
        }

        public virtual float this[int i]
        {
            get
            {
                return 0;
            }
            set
            {
                value = 0;
            }
        }
    }


    public class CombinedVector : VectorBase
    {
        private List<SingleVector> m_innerData;
        int m_nLenPerBlock;
        int m_nLen;

        public override int GetDimension() { return m_nLen; }

        public CombinedVector()
        {
            m_innerData = new List<SingleVector>();
            m_nLen = 0;
            m_nLenPerBlock = 0;
        }

        public void Append(SingleVector vector)
        {
            if (m_nLenPerBlock > 0)
            {
                if (m_nLenPerBlock != vector.GetDimension())
                {
                    throw new Exception("The dimension of appending vector is not the same as the previous one");
                }
            }

            m_innerData.Add(vector);
            m_nLenPerBlock = vector.GetDimension();
            m_nLen += m_nLenPerBlock;
        }


        public override float this[int i]
        {
            get
            {
                return m_innerData[i / m_nLenPerBlock][i % m_nLenPerBlock];
            }
            set
            {
                m_innerData[i / m_nLenPerBlock][i % m_nLenPerBlock] = value;
            }
        }
    }


    public class SingleVector : VectorBase
    {
        private float[] m_innerData;
        int m_nLen;
        public override int GetDimension() { return m_nLen; }

        public SingleVector()
        {
            m_innerData = null;
        }

        public SingleVector(int nLen, float[] val)
        {
            m_nLen = nLen;
            m_innerData = new float[m_nLen];
            for (int i = 0; i < m_nLen; i++)
            {
                m_innerData[i] = val[i];
            }
        }

        public SingleVector(int nLen)
        {
            m_innerData = new float[nLen];
            m_nLen = nLen;
        }


        public override float this[int i]
        {
            get
            {
                return m_innerData[i];
            }
            set
            {
                m_innerData[i] = value;
            }
        }
    }
}
