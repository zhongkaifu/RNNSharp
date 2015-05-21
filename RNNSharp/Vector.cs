using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class Vector
    {
        public virtual int GetDimension()
        {
            return 0;
        }

        public virtual double this[int i]
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


    public class CombinedVector : Vector
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


        public override double this[int i]
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


    public class SingleVector : Vector
    {
        private double[] m_innerData;
        int m_nLen;
        public override int GetDimension() { return m_nLen; }

        public SingleVector()
        {
            m_innerData = null;
        }

        public SingleVector(int nLen, double[] val)
        {
            m_nLen = nLen;
            m_innerData = new double[m_nLen];
            for (int i = 0; i < m_nLen; i++)
            {
                m_innerData[i] = val[i];
            }
        }

        public SingleVector(int nLen)
        {
            m_innerData = new double[nLen];
            m_nLen = nLen;
        }


        public override double this[int i]
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


        public SingleVector Set(SingleVector rhs, int startOffset)
        {
            for (int i = 0; i < rhs.GetDimension(); i++)
            {
                m_innerData[i + startOffset] = rhs.m_innerData[i];
            }
            return this;
        }

        public void Normalize()
        {

            double sum = 0;
            for (int i = 0; i < m_nLen; i++)
            {
                sum += m_innerData[i] * m_innerData[i];
            }

            if (0 == sum) return;
            double df = Math.Sqrt(sum);

            for (int i = 0; i < m_nLen; i++)
            {
                m_innerData[i] /= df;
            }
        }
    }
}
