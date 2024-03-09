# 将远程服务器root@172.17.0.2的debug_copy3更新并下载到本地的"temp\$locName\$dateTime"路径下

param (
    [string]$locName
)


# 远程控制远程机器执行复制文件
$remoteHostName = "172.17.0.2"
$remoteUserName = "root"
$remoteBasePath = "/usr/local/rvmount/appdata"
$remoteDataDir = "debug_copy"
route add 172.17.0.0 mask 255.255.255.0 172.16.255.1
$date = Get-Date -Format "yyyy-MM-dd" #获取捕获日期
$dateTime = Get-Date -Format "yyyyMMdd_HHmmss" #保存捕获时间
ssh $remoteUserName@$remoteHostName "cd $remoteBasePath ; rm -rf $remoteDataDir; mkdir $remoteDataDir;  cp debug/sti*.jpg $remoteDataDir/;cp debug/STI_MOT*.jpg $remoteDataDir/;cp result/$date/flow_speed_evaluation_result.csv $remoteDataDir/; cd $remoteDataDir && ls;exit;"

# 远程数据路径
$remoteDataPath = "$remoteBasePath/$remoteDataDir"

# 移动执行路径到项目根目录
$scriptPath = $MyInvocation.MyCommand.Path
$scriptDirectory = Split-Path -Path $scriptPath -Parent
Set-Location "$scriptDirectory\.."

#本地下载，如果下载成功复制到$localDataDir下
$localDataDir = "data\data_new"
$localDataPath = "$localdataDir\$locName\$dateTime"
$localTempPath = "deprecated\data\temp"
while ($true) {
    # 删除已下载的目录
    Remove-Item -Path $localTempPath -Recurse -Force | Out-Null

    # 创建本地目录
    New-Item -ItemType Directory -Path $localTempPath | Out-Null
    
    # 下载
    sftp -r $remoteUserName@${remoteHostName}:$remoteDataPath $localTempPath

    # 检查下载结果
    if ($?) {
        Write-Host "download successed"
        break
    }
    else {
        Write-Host "download failed"
        Start-Sleep -Seconds 5
    }
}

# 移动文件
New-Item -ItemType Directory -Path $localDataPath | Out-Null
Get-ChildItem $localTempPath | Move-Item -Destination $localDataPath
