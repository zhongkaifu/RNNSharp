using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class WordEMWrapFeaturizer
    {
        public int vectorSize;
        public Dictionary<string, SingleVector> m_WordEmbedding;
        public SingleVector m_UnkEmbedding;

        public WordEMWrapFeaturizer(string filename, bool textFormat = false)
        {
            Txt2Vec.Model model = new Txt2Vec.Model();
            model.LoadModel(filename, textFormat);

            string[] terms = model.GetAllTerms();
            vectorSize = model.VectorSize;

            m_WordEmbedding = new Dictionary<string, SingleVector>();
            m_UnkEmbedding = new SingleVector(vectorSize);

            foreach (string term in terms)
            {
                float[] vector = model.GetVector(term);

                if (vector != null)
                {
                    SingleVector spVector = new SingleVector(vectorSize, vector);
                    m_WordEmbedding.Add(term, spVector);
                }
            }
        }

        public int GetDimension()
        {
            return vectorSize;
        }

        public SingleVector GetTermVector(string strTerm)
        {
            if (m_WordEmbedding.ContainsKey(strTerm) == true)
            {
                return m_WordEmbedding[strTerm];
            }

            return m_UnkEmbedding;
        }
    }
}
