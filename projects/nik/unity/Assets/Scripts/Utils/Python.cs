// Numenta Platform for Intelligent Computing (NuPIC)
// Copyright (C)  2016, Numenta, Inc.  Unless you have an agreement
// with Numenta, Inc., for a separate license for this software code, the
// following terms and conditions apply:
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero Public License version 3 as
// published by the Free Software Foundation.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Affero Public License for more details.
// You should have received a copy of the GNU Affero Public License
// along with this program.  If not, see http://www.gnu.org/licenses.
// http://numenta.org/licenses/

using UnityEngine;
using System.IO;
using System.Diagnostics;
using System;

namespace Numenta.Utils
{
    /// <summary>
    /// Spawn python process
    /// </summary>
    public class Python
    {
        protected Process process;
        public String ErrorLog { get; private set; }

        public bool EnableLogging = false;
        private StreamWriter LogStream;

        /// <summary>
        /// Starts a python process and runs the given script using the python
        // interpreter installed on the user computer.
        /// Python scripts are stored as <see cref="TextAsset"/> in the
        /// "Assets/Resources/Python" directory. See <see cref="Resources"/>
        /// </summary>
        /// <param name="scriptName">Python script name</param>
        /// <param name="args">Optional arguments</param>
        public Python(string scriptName, string args = null)
        {
            // Copy TextAsset from resource bundle to disk so it can be executed
            string pythonScriptsPath = Path.Combine(Application.persistentDataPath, "python");

            // Make sure directory exists
            if (!Directory.Exists(pythonScriptsPath))
            {
                Directory.CreateDirectory(pythonScriptsPath);
            }
            string scriptPath = Path.Combine(pythonScriptsPath, scriptName);
            if (!File.Exists(scriptPath))
            {
                string assetPath = Path.Combine("Python", scriptName);
                TextAsset asset = Resources.Load(assetPath) as TextAsset;
                File.WriteAllText(scriptPath, asset.ToString());
            }

            // Configure Python process
            process = new Process();
            process.EnableRaisingEvents = true;
            process.StartInfo.FileName = "python";
            process.StartInfo.Arguments = "'" + @scriptPath + "'";
            process.StartInfo.WorkingDirectory = Application.dataPath;
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.CreateNoWindow = true;
            process.StartInfo.RedirectStandardOutput = true;
            process.StartInfo.RedirectStandardInput = true;
            process.StartInfo.RedirectStandardError = true;
            process.ErrorDataReceived += (sender, e) =>
            {
                if (!String.IsNullOrEmpty(e.Data))
                {
                    UnityEngine.Debug.LogError(e.Data);
                    ErrorLog += e.Data;
                    if (LogStream != null)
                    {
                        var message = string.Format("[{0:H:mm:ss}] ERROR {1}", DateTime.Now,  e.Data);
                        LogStream.WriteLine(message);
                        LogStream.Flush();
                    }
                }
            };
            AddArguments(args);
        }
        protected virtual void AddArguments(string arguments)
        {
            if (arguments != null)
            {
                process.StartInfo.Arguments += " " + @arguments;
            }
        }
        /// <summary>
        /// Starts the python process asynchronously
        /// </summary>
        public virtual void Start()
        {
            // Log
            string logPath = Path.Combine(Application.persistentDataPath, "log");
            // Make sure directory exists
            if (!Directory.Exists(logPath))
            {
                Directory.CreateDirectory(logPath);
            }
            if (EnableLogging)
            {
                LogStream = new StreamWriter(Path.Combine(logPath, "python.log"), true);
                var message = string.Format("[{0:H:mm:ss}] INFO {1} {2}",
                                            DateTime.Now, "Started",
                                            process.StartInfo.Arguments);
                LogStream.WriteLine(message);
                LogStream.Flush();
            }

            process.Start();
            process.BeginErrorReadLine();
        }

        /// <summary>
        /// Stops the asynchronous process
        /// </summary>
        public virtual void Stop()
        {
            process.Close();
            if (LogStream != null)
            {
                var message = string.Format("[{0:H:mm:ss}] INFO {1} {2}",
                                            DateTime.Now, "Stopped",
                                            process.StartInfo.Arguments);
                LogStream.WriteLine(message);
                LogStream.Flush();

                LogStream.Close();
                LogStream = null;
            }
        }
        /// <summary>
        /// Writes a string to the process stdin
        /// </summary>
        /// <param name="data"></param>
        public void Write(string data)
        {
            if (LogStream != null)
            {
                var message = string.Format("[{0:H:mm:ss}] INFO Write {1}", DateTime.Now, data);
                LogStream.WriteLine(message);
                LogStream.Flush();
            }

            // UnityEngine.Debug.Log(data);
            // FIXME: .NET StreamWriter prepends stream with UTF-8 BOM.
            // To avoid problems with python interpreter (https://bugs.python.org/issue21927)
            // we are writing the encoded bytes directly to the stream skipping the StreamWriter
            var bytes = process.StandardInput.Encoding.GetBytes(data);
            process.StandardInput.BaseStream.Write(bytes, 0, bytes.Length);
            process.StandardInput.BaseStream.Flush();
        }

        /// <summary>
        /// Writes a string followed by a line terminator to the process stdin
        /// </summary>
        /// <param name="line"></param>
        public void WriteLine(string line = "")
        {
            Write(line + System.Environment.NewLine);
        }

        public string ReadToEnd()
        {
            var line = process.StandardOutput.ReadToEnd();
            if (LogStream != null)
            {
                var message = string.Format("[{0:H:mm:ss}] INFO Read {1}", DateTime.Now, line);
                LogStream.WriteLine(message);
                LogStream.Flush();
            }
            return line;
        }
        public string ReadLine()
        {
            var line = process.StandardOutput.ReadLine();
            if (LogStream != null)
            {
                var message = string.Format("[{0:H:mm:ss}] INFO Read {1}", DateTime.Now, line);
                LogStream.WriteLine(message);
                LogStream.Flush();
            }
            return line;
        }
        public bool IsRunning
        {
            get
            {
                return process != null && !process.HasExited;
            }
        }
        public void ClearErrorLog()
        {
            ErrorLog = "";
        }
    }
}
