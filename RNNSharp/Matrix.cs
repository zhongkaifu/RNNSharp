/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class Matrix<T> where T : struct
    {
        private readonly T[][] m_saData;

        public Matrix(int h, int w)
        {
            Height = h;
            Width = w;
            m_saData = new T[Height][];
            for (var i = 0; i < Height; i++)
            {
                m_saData[i] = new T[Width];
            }
        }

        public int Height { get; set; } // the number of rows
        public int Width { get; set; } // the number of columns

        public T[] this[int i]
        {
            get { return m_saData[i]; }
            set { m_saData[i] = value; }
        }
    }
}