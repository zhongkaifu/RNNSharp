using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace RNNSharp
{
    public class Matrix
    {

        private int m_nHeight; // the number of rows
        private int m_nWidth; // the number of columns
        private double[][] m_saData;

        public static ParallelOptions parallelOption = new ParallelOptions();

        public void Reset()
        {
            for (int i = 0; i < m_saData.Length; i++)
            {
                Array.Clear(m_saData[i], 0, m_saData[i].Length);
            }
        }


        public Matrix Add(Matrix mat)
        {
            Parallel.For(0, m_nHeight, parallelOption, i =>
            {
                for (int j = 0; j < m_nWidth; j++)
                {
                    m_saData[i][j] += mat.m_saData[i][j];
                }
            });
            return this;
        }

        public Matrix Scale(double factor)
        {
            Parallel.For(0, m_nHeight, parallelOption, i =>
            {
                for (int j = 0; j < m_nWidth; j++)
                    m_saData[i][j] *= factor;
            });

            return this;
        }

        void Alloc()
        {
            m_saData = new double[m_nHeight][];
            for (int i = 0; i < m_nHeight; i++)
            {
                m_saData[i] = new double[m_nWidth];
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


        public static Matrix Sub(Matrix mat1, Matrix mat2)
        {
            Matrix mat = new Matrix(mat1.m_nHeight, mat1.m_nWidth);

            Parallel.For(0, mat.m_nHeight, parallelOption, i =>
            {
                for (int j = 0; j < mat.m_nWidth; j++)
                {
                    mat.m_saData[i][j] = mat1.m_saData[i][j] - mat2.m_saData[i][j];
                }
            });

            return mat;
        }

        public double[] this[int i]
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

        public void Deserialize(BinaryReader fin)
        {

            int nWidth, nHeight;

            nWidth = fin.ReadInt32();
            nHeight = fin.ReadInt32();

            Deserialize(nHeight, nWidth, fin);
        }

        //Read matrix data from given reader
        void Deserialize(int h, int w, BinaryReader fin)
        {
            m_nHeight = h;
            m_nWidth = w;
            Alloc();
            for (int i = 0; i < m_nHeight; i++)
            {
                for (int j = 0; j < m_nWidth; j++)
                {
                    m_saData[i][j] = fin.ReadSingle();
                }
            }
        }

    }
}
