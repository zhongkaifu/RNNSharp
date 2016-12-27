using AdvUtils;
using System;
using System.Collections.Generic;
using System.Text;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class Sentence
    {
        public Sentence(List<string[]> tokensList, bool addSentBE = true)
        {
            var dim = 0;
            TokensList = new List<string[]>();

            if (tokensList.Count == 0)
            {
                return;
            }

            //Check if dimension is consistent inside the sentence
            foreach (var tokens in tokensList)
            {
                if (dim > 0 && tokens.Length != dim)
                {
                    var err = ReportInvalidateTokens(tokensList, dim, tokens);
                    throw new FormatException($"Invalidated record: {err}");
                }

                dim = tokens.Length;
                TokensList.Add(tokens);
            }

            if (addSentBE)
            {
                //Add begin/end of sentence flag into feature
                var beginFeatures = new string[dim];
                var endFeatures = new string[dim];

                for (var i = 0; i < dim - 1; i++)
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

        public List<string[]> TokensList { get; }

        public override string ToString()
        {
            var sb = new StringBuilder();
            foreach (var tokens in TokensList)
            {
                foreach (var token in tokens)
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
            var sb = new StringBuilder();
            sb.AppendLine(
                $"Inconsistent feature dimension in the record.It's {badTokens.Length}, but it should be {dim}");
            sb.AppendLine(ToString());
            Logger.WriteLine(Logger.Level.err, sb.ToString());

            return sb.ToString();
        }
    }
}