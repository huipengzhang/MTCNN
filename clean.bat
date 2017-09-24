@echo off
del windows\*.lock
del bin\*.pdb
if exist windows\Release rd windows\Release /s /q
if exist MTCNN-light\x64 rd MTCNN-light\x64 /s /q
pause