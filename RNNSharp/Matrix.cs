
/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class Matrix<T>
    {

        private int m_nHeight; // the number of rows
        private int m_nWidth; // the number of columns
        private T[][] m_saData;

        public Matrix(int h, int w)
        {
            m_nHeight = h;
            m_nWidth = w;
            m_saData = new T[m_nHeight][];
            for (int i = 0; i < m_nHeight; i++)
            {
                m_saData[i] = new T[m_nWidth];
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

        public int GetWidth()
        {
            return m_nWidth;
        }

        public int GetHeight()
        {
            return m_nHeight;
        }

        public Matrix<T> CopyTo()
        {
            Matrix<T> m = new Matrix<T>(m_nHeight, m_nWidth);

            for (int i = 0; i < m_nHeight; i++)
            {
                for (int j = 0; j < m_nWidth; j++)
                {
                    m[i][j] = m_saData[i][j];
                }
            }

            return m;
        }
    }
}
