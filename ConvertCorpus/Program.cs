using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using AdvUtils;

namespace ConvertCorpus
{
    class Program
    {
        static int ArgPos(string str, string[] args)
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
            StreamReader sr = new StreamReader(strInputFile);
            StreamWriter sw = new StreamWriter(strOutputFile);
            string strLine = null;

            while ((strLine = sr.ReadLine()) != null)
            {
                strLine = strLine.Trim();

                string[] items = strLine.Split();
                foreach (string item in items)
                {
                    int pos = item.LastIndexOf('[');
                    string strTerm = item.Substring(0, pos);
                    string strTag = item.Substring(pos + 1, item.Length - pos - 2);

                    sw.WriteLine("{0}\t{1}", strTerm, strTag);
                }
                sw.WriteLine();
            }

            sr.Close();
            sw.Close();
        }

        static void Main(string[] args)
        {
            ConvertFormat(args[0], args[1]);
        }
    }
}
