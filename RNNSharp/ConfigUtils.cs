using System;
using System.Collections.Generic;
using System.IO;

namespace RNNSharp
{
    public class ConfigUtils
    {
        private string configFilePath;
        private Dictionary<string, string> items;

        public void LoadFile(string configFile)
        {
            configFilePath = configFile;
            items = new Dictionary<string, string>();
            using (var sr = new StreamReader(configFile))
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    line = line.Trim().ToLower();
                    if (line.Length == 0)
                    {
                        //Emtpy line, ignore it
                        continue;
                    }

                    if (line.StartsWith("#"))
                    {
                        //Comments line, ignore it
                        continue;
                    }

                    var pos = line.IndexOf('=');
                    var key = line.Substring(0, pos).Trim();
                    var value = line.Substring(pos + 1).Trim();

                    items.Add(key, value);
                }
            }
        }

        public string GetValueOptional(string key)
        {
            string value;
            return items.TryGetValue(key.ToLower().Trim(), out value) ? value : null;
        }

        public string GetValueRequired(string key)
        {
            string value;
            if (items.TryGetValue(key.ToLower().Trim(), out value))
            {
                return value;
            }
            throw new ArgumentNullException($"Fail to get '{key}' value from '{configFilePath}'.");
        }
    }
}