using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace RNNSharp
{
    public class Matrix<T>
    {

        private int m_nHeight; // the number of rows
        private int m_nWidth; // the number of columns
        private T[][] m_saData;

        public void Reset()
        {
            Alloc();
        }

        void Alloc()
        {
            m_saData = new T[m_nHeight][];
            for (int i = 0; i < m_nHeight; i++)
            {
                m_saData[i] = new T[m_nWidth];
            }

        }

        public Matrix()
        {

        }

        public Matrix(int h, int w)
        {
            m_nHeight = h;
            m_nWidth = w;
            Alloc();
        }


        public T[] this[int i]
        {
            get
            {
                return m_saData[i];
            }
            set
            {
                m_saData[i] = value;
            }
        }

        public int GetWidth()
        {
            return m_nWidth;
        }

        public int GetHeight()
        {
            return m_nHeight;
        }
    }
}
