using System.Numerics;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class Matrix<T> where T : struct
    {

        public int Height { get; set; } // the number of rows
        public int Width { get; set; } // the number of columns
        private T[][] m_saData;

        public Matrix(int h, int w)
        {
            Height = h;
            Width = w;
            m_saData = new T[Height][];
            for (int i = 0; i < Height; i++)
            {
                m_saData[i] = new T[Width];
            }
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

        public Matrix<T> CopyTo()
        {
            Matrix<T> m = new Matrix<T>(Height, Width);

            for (int i = 0; i < Height; i++)
            {
                T[] m_i = m[i];
                T[] m_saData_i = m_saData[i];
				int j = 0;
                while (j < Width - Vector<T>.Count)
                {
                    Vector<T> v1 = new Vector<T>(m_saData_i, j);
                    v1.CopyTo(m_i, j);

                    j += Vector<T>.Count;
                }

                while (j < Width)
                {
                    m_i[j] = m_saData_i[j];
                    j++;
                }
            }

            return m;
        }
    }
}
