using System.Collections.Generic;
using Txt2Vec;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class WordEMWrapFeaturizer
    {
        public SingleVector m_UnkEmbedding;
        public Dictionary<string, SingleVector> m_WordEmbedding;
        public int vectorSize;

        public WordEMWrapFeaturizer(string filename, bool textFormat = false)
        {
            var model = new Model();
            model.LoadModel(filename, textFormat);

            var terms = model.GetAllTerms();
            vectorSize = model.VectorSize;

            m_WordEmbedding = new Dictionary<string, SingleVector>();
            m_UnkEmbedding = new SingleVector(vectorSize);

            foreach (var term in terms)
            {
                var vector = model.GetVector(term);

                if (vector != null)
                {
                    var spVector = new SingleVector(vector);
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
            return m_WordEmbedding.ContainsKey(strTerm) ? m_WordEmbedding[strTerm] : m_UnkEmbedding;
        }
    }
}