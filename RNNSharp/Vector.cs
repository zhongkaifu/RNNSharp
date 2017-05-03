using System;
using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class VectorBase
    {
        public virtual int Length => 0;

        public virtual float this[int i]
        {
            get { return 0; }
            set { value = 0; }
        }

        public virtual float[] CopyTo()
        {
            return null;
        }
    }

    public class CombinedVector : VectorBase
    {
        private readonly List<SingleVector> m_innerData;
        private int m_nLen;
        private int m_nLenPerBlock;

        public CombinedVector()
        {
            m_innerData = new List<SingleVector>();
            m_nLen = 0;
            m_nLenPerBlock = 0;
        }

        public override int Length => m_nLen;

        public override float this[int i]
        {
            get { return m_innerData[i / m_nLenPerBlock][i % m_nLenPerBlock]; }
            set { m_innerData[i / m_nLenPerBlock][i % m_nLenPerBlock] = value; }
        }

        public void Append(SingleVector vector)
        {
            if (m_nLenPerBlock > 0)
            {
                if (m_nLenPerBlock != vector.Length)
                {
                    throw new Exception("The dimension of appending vector is not the same as the previous one");
                }
            }

            m_innerData.Add(vector);
            m_nLenPerBlock = vector.Length;
            m_nLen += m_nLenPerBlock;
        }

        public override float[] CopyTo()
        {
            var val = new float[m_nLen];
            for (var i = 0; i < m_innerData.Count; i++)
            {
                for (var j = 0; j < m_innerData[i].Length; j++)
                {
                    val[i * m_nLenPerBlock + j] = m_innerData[i][j];
                }
            }

            return val;
        }
    }

    public class SingleVector : VectorBase
    {
        private readonly float[] m_innerData;

        public SingleVector()
        {
            m_innerData = null;
        }

        public SingleVector(float[] val)
        {
            m_innerData = val;
        }

        public SingleVector(int nLen)
        {
            m_innerData = new float[nLen];
        }

        public override int Length
        {
            get
            {
                if (m_innerData == null)
                {
                    return 0;
                }
                else
                {
                    return m_innerData.Length;
                }
            }
        }

        public override float this[int i]
        {
            get { return m_innerData[i]; }
            set { m_innerData[i] = value; }
        }

        public override float[] CopyTo()
        {
            return m_innerData;
        }
    }
}