# 将远程服务器root@172.17.0.2的debug_copy3更新并下载到本地的"temp3\$locName\$dateTime"路径下

param (
    [string]$locName
)

Set-Location D:\Documents\Code\flood2

$hostName = "172.17.0.2"
$userName = "root"
$remoteBaseDir = "/usr/local/rvmount/appdata/"
$dirName = "debug_copy"
$remoteDir = $remoteBaseDir + $dirName

route add 172.17.0.0 mask 255.255.255.0 172.16.255.1
ssh $userName@$hostName "cd $remoteBaseDir ; rm -rf $dirName; mkdir $dirName; mkdir $dirName/hwMot; cp debug/sti*.jpg $dirName/;cp debug/STI_MOT*.jpg $dirName/hwMot; cd $dirName && ls;exit;"

$dateTime = Get-Date -Format "yyyyMMdd_HHmmss"
$localDir = "temp\$locName\$dateTime"
$localDir_ = "deprecated\data\temp"
while ($true) {
    # 删除已下载的目录
    Remove-Item -Path $localDir_ -Recurse -Force

    # 创建本地目录
    New-Item -ItemType Directory -Path $localDir_ | Out-Null
    
    sftp -r $userName@${hostName}:$remoteDir $localDir_

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

New-Item -ItemType Directory -Path $localDir | Out-Null
Get-ChildItem $localDir_ | Move-Item -Destination $localDir
