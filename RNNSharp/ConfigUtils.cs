using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace RNNSharp
{
    public class ConfigUtils
    {
        string configFilePath;
        Dictionary<string, string> items;
        public void LoadFile(string configFile)
        {
            configFilePath = configFile;
            items = new Dictionary<string, string>();
            using (StreamReader sr = new StreamReader(configFile))
            {
                string line = String.Empty;
                while ((line = sr.ReadLine()) != null)
                {
                    line = line.Trim().ToLower();
                    if (line.Length == 0)
                    {
                        //Emtpy line, ignore it
                        continue;
                    }

                    if (line.StartsWith("#") == true)
                    {
                        //Comments line, ignore it
                        continue;
                    }

                    int pos = line.IndexOf('=');
                    string key = line.Substring(0, pos).Trim();
                    string value = line.Substring(pos + 1).Trim();

                    items.Add(key, value);
                }
            }
        }

        public string GetValueOptional(string key)
        {
            string value = String.Empty;
            if (items.TryGetValue(key.ToLower().Trim(), out value))
            {
                return value;
            }

            return null;
        }

        public string GetValueRequired(string key)
        {
            string value = String.Empty;
            if (items.TryGetValue(key.ToLower().Trim(), out value))
            {
                return value;
            }
            else
            {
                throw new ArgumentNullException($"Fail to get '{key}' value from '{configFilePath}'.");
            }
        }
    }
}
