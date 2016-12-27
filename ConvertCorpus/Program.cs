using AdvUtils;
using System.IO;

namespace ConvertCorpus
{
    internal class Program
    {
        private static int ArgPos(string str, string[] args)
        {
            int a;
            for (a = 0; a < args.Length; a++)
            {
                if (str == args[a])
                {
                    if (a == args.Length - 1)
                    {
                        Logger.WriteLine("Argument missing for {0}", str);
                        return -1;
                    }
                    return a;
                }
            }
            return -1;
        }

        public static void ConvertFormat(string strInputFile, string strOutputFile)
        {
            var sr = new StreamReader(strInputFile);
            var sw = new StreamWriter(strOutputFile);
            string strLine;

            while ((strLine = sr.ReadLine()) != null)
            {
                strLine = strLine.Trim();

                var items = strLine.Split();
                foreach (var item in items)
                {
                    var pos = item.LastIndexOf('[');
                    var strTerm = item.Substring(0, pos);
                    var strTag = item.Substring(pos + 1, item.Length - pos - 2);

                    sw.WriteLine("{0}\t{1}", strTerm, strTag);
                }
                sw.WriteLine();
            }

            sr.Close();
            sw.Close();
        }

        private static void Main(string[] args)
        {
            ConvertFormat(args[0], args[1]);
        }
    }
}