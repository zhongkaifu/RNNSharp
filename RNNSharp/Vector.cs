using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class Vector
    {
        private double[] m_innerData;

        int m_nLen;
        static ParallelOptions parallelOption = new ParallelOptions();


        public virtual int GetDimension() { return m_nLen; }

        public Vector()
        {
            m_innerData = null;
        }

        public Vector(int nLen, double[] val)
        {
            m_nLen = nLen;
            m_innerData = new double[m_nLen];
            for (int i = 0; i < m_nLen; i++)
            {
                m_innerData[i] = val[i];
            }
        }

        public Vector(int nLen)
        {
            m_innerData = new double[nLen];
            m_nLen = nLen;
        }


        public double this[int i]
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


        public Vector Set(Vector rhs, int startOffset)
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
