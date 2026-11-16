##Returns the names of the last 5 log files created
Get-ChildItem .\logs\*.jsonl |                                              
>>     Sort-Object LastWriteTime -Descending |
>>     Select-Object -First 5 Name, LastWriteTime

##Sets the value of the maximum number of steps to x
$env:AGENT_MAX_STEPS="x"  

##Checks the last output for errors using the log file and counts the model steps
Get-Content -Encoding UTF8 $file |
>>     ConvertFrom-Json |
>>     Where-Object { $_.event -in @("final", "error") } |
>>     Select-Object event, ok, steps, reason, error