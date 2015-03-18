using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class WordEMWrapFeaturizer
    {
        public int vectorSize;
        public Dictionary<string, Vector> m_WordEmbedding;
        public Vector m_UnkEmbedding;

        public WordEMWrapFeaturizer(string filename)
        {
            Txt2Vec.Decoder decoder = new Txt2Vec.Decoder();
            decoder.LoadBinaryModel(filename);

            string[] terms = decoder.GetAllTerms();
            vectorSize = decoder.GetVectorSize();

            m_WordEmbedding = new Dictionary<string, Vector>();
            m_UnkEmbedding = new Vector(vectorSize);

            foreach (string term in terms)
            {
                double[] vector = decoder.GetVector(term);

                if (vector != null)
                {
                    Vector spVector = new Vector(vectorSize, vector);

                    spVector.Normalize();

                    m_WordEmbedding.Add(term, spVector);
                }
            }
        }

        public int GetDimension()
        {
            return vectorSize;
        }

        public Vector GetTermVector(string strTerm)
        {
            if (m_WordEmbedding.ContainsKey(strTerm) == true)
            {
                return m_WordEmbedding[strTerm];
            }

            return m_UnkEmbedding;
        }
    }
}
