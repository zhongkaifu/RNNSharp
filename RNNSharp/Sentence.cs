using System;
using System.Collections.Generic;
using System.Text;
using AdvUtils;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class Sentence
    {
        public List<string[]> TokensList { get; }

        public Sentence(List<string[]> tokensList, bool addSentBE = true)
        {
            int dim = 0;
            TokensList = new List<string[]>();

            if (tokensList.Count == 0)
            {
                return;
            }

            //Check if dimension is consistent inside the sentence
            foreach (string[] tokens in tokensList)
            {
                if (dim > 0 && tokens.Length != dim)
                {
                    string err = ReportInvalidateTokens(tokensList, dim, tokens);
                    throw new FormatException(String.Format("Invalidated record: {0}", err));
                }

                dim = tokens.Length;
                TokensList.Add(tokens);
            }

            if (addSentBE)
            {
                //Add begin/end of sentence flag into feature
                string[] beginFeatures = new string[dim];
                string[] endFeatures = new string[dim];

                for (int i = 0; i < dim - 1; i++)
                {
                    beginFeatures[i] = "<s>";
                    endFeatures[i] = "</s>";
                }

                beginFeatures[dim - 1] = TagSet.DefaultTag;
                endFeatures[dim - 1] = TagSet.DefaultTag;

                TokensList.Insert(0, beginFeatures);
                TokensList.Add(endFeatures);
            }
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            foreach (string[] tokens in TokensList)
            {           
                foreach (string token in tokens)
                {
                    sb.Append(token);
                    sb.Append('\t');
                }
                sb.AppendLine();
            }

            return sb.ToString();
        }

        private string ReportInvalidateTokens(List<string[]> tokenList, int dim, string[] badTokens)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(String.Format("Inconsistent feature dimension in the record.It's {0}, but it should be {1}", badTokens.Length, dim));
            sb.AppendLine(ToString());
            Logger.WriteLine(Logger.Level.err, sb.ToString());

            return sb.ToString();
        }
    }
}
