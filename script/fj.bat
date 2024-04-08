@echo off
:loop
PowerShell C:\BaseDir\code\flood\script\download.ps1 -locName fj
timeout /t 3600 >nul
goto loop
