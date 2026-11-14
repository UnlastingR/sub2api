<#
.SYNOPSIS
    交互式安全保存并调用 sub2api Scoped Admin Key (adm_sc_...) 的本地管理脚本。

.DESCRIPTION
    使用 Windows Credential Manager (Advapi32 CredWriteW / CredReadW / CredDeleteW)
    以 DPAPI 加密保存 adm_sc_... 控制密钥（不落明文文件、不进环境变量、不留命令行历史），
    并通过 x-api-key 请求头调用 /api/v1/admin/... 管理接口。

.EXAMPLE
    # 启动交互式菜单
    .\tools\sub2api-admin.ps1

.EXAMPLE
    # 非交互直接调用接口（密钥仍从 Windows 凭据管理器读取）
    .\tools\sub2api-admin.ps1 -Action call -Method GET -Path /api/v1/admin/system/version
#>
[CmdletBinding()]
param(
    [ValidateSet('menu', 'save', 'status', 'call', 'delete')]
    [string]$Action = 'menu',

    [string]$BaseUrl = 'https://sub2api.fyzure.fyi',

    [string]$Target = '',

    [ValidateSet('GET', 'POST', 'PUT', 'PATCH', 'DELETE')]
    [string]$Method = 'GET',

    [string]$Path = '',

    [string]$Body = '',

    [string]$BodyFile = '',

    [int]$TimeoutSec = 30,

    [switch]$Raw
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

try {
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
    # Ignore console encoding adjustments in non-interactive hosts.
}

if (-not ('Sub2Api.WinCredManager' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

namespace Sub2Api
{
    public static class WinCredManager
    {
        private const uint CRED_TYPE_GENERIC = 1;
        private const uint CRED_PERSIST_LOCAL_MACHINE = 2;

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct CREDENTIAL
        {
            public uint Flags;
            public uint Type;
            public string TargetName;
            public string Comment;
            public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
            public uint CredentialBlobSize;
            public IntPtr CredentialBlob;
            public uint Persist;
            public uint AttributeCount;
            public IntPtr Attributes;
            public string TargetAlias;
            public string UserName;
        }

        [DllImport("advapi32.dll", EntryPoint = "CredReadW", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool CredRead(string target, uint type, uint reservedFlag, out IntPtr credentialPtr);

        [DllImport("advapi32.dll", EntryPoint = "CredWriteW", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool CredWrite([In] ref CREDENTIAL userCredential, [In] uint flags);

        [DllImport("advapi32.dll", EntryPoint = "CredDeleteW", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool CredDelete(string target, uint type, uint flags);

        [DllImport("advapi32.dll", EntryPoint = "CredFree", SetLastError = true)]
        private static extern void CredFree([In] IntPtr buffer);

        public static void WriteUtf8Secret(string targetName, string userName, byte[] secretBytes, string comment)
        {
            if (string.IsNullOrWhiteSpace(targetName))
            {
                throw new ArgumentException("TargetName cannot be empty.");
            }
            if (secretBytes == null || secretBytes.Length == 0)
            {
                throw new ArgumentException("Secret cannot be empty.");
            }

            IntPtr blob = Marshal.AllocHGlobal(secretBytes.Length);
            try
            {
                Marshal.Copy(secretBytes, 0, blob, secretBytes.Length);
                var cred = new CREDENTIAL
                {
                    Type = CRED_TYPE_GENERIC,
                    TargetName = targetName,
                    UserName = string.IsNullOrWhiteSpace(userName) ? "sub2api-admin" : userName,
                    Comment = comment ?? string.Empty,
                    CredentialBlobSize = (uint)secretBytes.Length,
                    CredentialBlob = blob,
                    Persist = CRED_PERSIST_LOCAL_MACHINE
                };

                if (!CredWrite(ref cred, 0))
                {
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                }
            }
            finally
            {
                // Zero unmanaged heap buffer before freeing.
                for (int i = 0; i < secretBytes.Length; i++)
                {
                    Marshal.WriteByte(blob, i, 0);
                }
                Marshal.FreeHGlobal(blob);
            }
        }

        public static byte[] ReadUtf8SecretBytes(string targetName)
        {
            IntPtr credPtr;
            if (!CredRead(targetName, CRED_TYPE_GENERIC, 0, out credPtr))
            {
                int err = Marshal.GetLastWin32Error();
                if (err == 1168) // ERROR_NOT_FOUND
                {
                    return null;
                }
                throw new Win32Exception(err);
            }

            try
            {
                var cred = Marshal.PtrToStructure<CREDENTIAL>(credPtr);
                if (cred.CredentialBlob == IntPtr.Zero || cred.CredentialBlobSize == 0)
                {
                    return Array.Empty<byte>();
                }
                var bytes = new byte[cred.CredentialBlobSize];
                Marshal.Copy(cred.CredentialBlob, bytes, 0, (int)cred.CredentialBlobSize);
                return bytes;
            }
            finally
            {
                CredFree(credPtr);
            }
        }

        public static bool DeleteSecret(string targetName)
        {
            if (!CredDelete(targetName, CRED_TYPE_GENERIC, 0))
            {
                int err = Marshal.GetLastWin32Error();
                if (err == 1168) // ERROR_NOT_FOUND
                {
                    return false;
                }
                throw new Win32Exception(err);
            }
            return true;
        }
    }
}
'@
}

try {
    Add-Type -AssemblyName System.Net.Http -ErrorAction Stop
} catch {
    # PowerShell 7 loads System.Net.Http automatically.
}

try {
    # Ensure TLS 1.2+ on Windows PowerShell 5.1.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {
    # Ignore on platforms where ServicePointManager is restricted.
}

function Resolve-NormalizedBaseUrl {
    param([string]$Url)
    $trimmed = $Url.Trim().TrimEnd('/')
    $uri = $null
    if (-not [Uri]::TryCreate($trimmed, [UriKind]::Absolute, [ref]$uri) -or ($uri.Scheme -notin @('https', 'http'))) {
        throw "无效的 BaseUrl: $Url（必须为 http:// 或 https:// 开头的绝对地址）"
    }
    if ($uri.Scheme -ne 'https' -and $uri.Host -notin @('127.0.0.1', 'localhost', '::1')) {
        Write-Warning "当前 BaseUrl 未使用 HTTPS ($trimmed)，控制密钥存在传输暴露风险。"
    }
    return $trimmed
}

function Resolve-CredentialTarget {
    param(
        [string]$NormalizedBaseUrl,
        [string]$ExplicitTarget
    )
    if (-not [string]::IsNullOrWhiteSpace($ExplicitTarget)) {
        return $ExplicitTarget.Trim()
    }
    $uri = [Uri]$NormalizedBaseUrl
    return "sub2api:admin-sc-key:$($uri.Authority)"
}

function Get-MaskedKeyPreview {
    param([string]$PlainKey)
    if ([string]::IsNullOrEmpty($PlainKey)) {
        return '<empty>'
    }
    if ($PlainKey.Length -le 12) {
        return ('*' * $PlainKey.Length)
    }
    $parts = $PlainKey.Split('_', 4)
    if ($parts.Length -eq 4 -and $parts[0] -eq 'adm' -and $parts[1] -eq 'sc') {
        $tail = $parts[3]
        $last4 = if ($tail.Length -ge 4) { $tail.Substring($tail.Length - 4) } else { $tail }
        return "adm_sc_$($parts[2])_****$last4"
    }
    return "$($PlainKey.Substring(0, 7))****$($PlainKey.Substring($PlainKey.Length - 4))"
}

function Convert-SecureStringToTrimmedUtf8Bytes {
    param([Security.SecureString]$SecureValue)
    $bstr = [IntPtr]::Zero
    try {
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)
        $raw = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        $trimmed = $raw.Trim()
        return [System.Text.Encoding]::UTF8.GetBytes($trimmed)
    }
    finally {
        if ($bstr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
    }
}

function Save-ControlKeyInteractive {
    param(
        [string]$TargetName,
        [string]$NormalizedBaseUrl
    )
    Write-Host ""
    Write-Host "凭据目标: $TargetName" -ForegroundColor Cyan
    Write-Host "站点地址: $NormalizedBaseUrl" -ForegroundColor Cyan
    Write-Host "请输入站点控制密钥 (输入过程不回显，格式如 adm_sc_<id>_<secret>):" -ForegroundColor Yellow

    $secureInput = Read-Host -AsSecureString
    $secretBytes = $null
    try {
        $secretBytes = Convert-SecureStringToTrimmedUtf8Bytes -SecureValue $secureInput
        if ($null -eq $secretBytes -or $secretBytes.Length -eq 0) {
            Write-Host "已取消：输入为空。" -ForegroundColor Red
            return $false
        }

        $preview = $null
        $plainCheck = [System.Text.Encoding]::UTF8.GetString($secretBytes)
        try {
            if (-not $plainCheck.StartsWith('adm_sc_', [StringComparison]::Ordinal)) {
                Write-Warning "密钥不是以 'adm_sc_' 开头，请确认是否粘贴完整。"
                $confirm = Read-Host "仍要保存该密钥吗? (y/N)"
                if ($confirm -notin @('y', 'Y', 'yes', 'YES')) {
                    Write-Host "已取消保存。" -ForegroundColor Yellow
                    return $false
                }
            }
            $preview = Get-MaskedKeyPreview -PlainKey $plainCheck
        }
        finally {
            $plainCheck = $null
        }

        $uri = [Uri]$NormalizedBaseUrl
        [Sub2Api.WinCredManager]::WriteUtf8Secret(
            $TargetName,
            $uri.Authority,
            $secretBytes,
            "sub2api scoped admin key for $NormalizedBaseUrl"
        )
        Write-Host "已安全写入 Windows 凭据管理器：$preview" -ForegroundColor Green
        return $true
    }
    finally {
        if ($null -ne $secretBytes) {
            [Array]::Clear($secretBytes, 0, $secretBytes.Length)
        }
        if ($null -ne $secureInput) {
            $secureInput.Dispose()
        }
    }
}

function Show-SavedKeyStatus {
    param(
        [string]$TargetName,
        [string]$NormalizedBaseUrl
    )
    $bytes = [Sub2Api.WinCredManager]::ReadUtf8SecretBytes($TargetName)
    if ($null -eq $bytes) {
        Write-Host "未找到已保存的控制密钥 (Target: $TargetName)" -ForegroundColor Yellow
        return $false
    }
    try {
        $plain = [System.Text.Encoding]::UTF8.GetString($bytes)
        try {
            $preview = Get-MaskedKeyPreview -PlainKey $plain
            Write-Host "已保存密钥状态:" -ForegroundColor Green
            Write-Host "  站点地址 : $NormalizedBaseUrl"
            Write-Host "  凭据目标 : $TargetName"
            Write-Host "  密钥摘要 : $preview (长度 $($bytes.Length) 字节)"
            return $true
        }
        finally {
            $plain = $null
        }
    }
    finally {
        [Array]::Clear($bytes, 0, $bytes.Length)
    }
}

function Remove-SavedKey {
    param([string]$TargetName)
    $deleted = [Sub2Api.WinCredManager]::DeleteSecret($TargetName)
    if ($deleted) {
        Write-Host "已从 Windows 凭据管理器删除：$TargetName" -ForegroundColor Green
    } else {
        Write-Host "未找到待删除的凭据：$TargetName" -ForegroundColor Yellow
    }
}

function Resolve-AdminApiPath {
    param([string]$InputPath)
    $p = $InputPath.Trim()
    if ([string]::IsNullOrWhiteSpace($p)) {
        throw "API Path 不能为空。"
    }
    if ($p.StartsWith('http://', [StringComparison]::OrdinalIgnoreCase) -or
        $p.StartsWith('https://', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path 请传相对路径（例如 /api/v1/admin/system/version 或 groups/all），不要传完整 URL。"
    }
    if (-not $p.StartsWith('/')) {
        $p = "/api/v1/admin/$p"
    } elseif (-not $p.StartsWith('/api/', [StringComparison]::OrdinalIgnoreCase)) {
        $p = "/api/v1/admin$p"
    }
    return $p
}

function Invoke-Sub2ApiAdminRequest {
    param(
        [string]$NormalizedBaseUrl,
        [string]$TargetName,
        [string]$HttpMethod,
        [string]$ApiPath,
        [string]$JsonBody = '',
        [string]$JsonBodyFile = '',
        [int]$RequestTimeoutSec = 30
    )

    if (-not [string]::IsNullOrWhiteSpace($JsonBodyFile)) {
        if (-not (Test-Path -LiteralPath $JsonBodyFile)) {
            throw "找不到 BodyFile: $JsonBodyFile"
        }
        $JsonBody = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $JsonBodyFile), [System.Text.Encoding]::UTF8)
    }

    $secretBytes = [Sub2Api.WinCredManager]::ReadUtf8SecretBytes($TargetName)
    if ($null -eq $secretBytes -or $secretBytes.Length -eq 0) {
        throw "未在 Windows 凭据管理器中找到密钥 ($TargetName)。请先运行保存流程 (-Action save)。"
    }

    $resolvedPath = Resolve-AdminApiPath -InputPath $ApiPath
    $fullUrl = "$NormalizedBaseUrl$resolvedPath"

    $handler = [System.Net.Http.HttpClientHandler]::new()
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds($RequestTimeoutSec)
    $request = $null
    $response = $null
    $plainKey = $null

    try {
        $plainKey = [System.Text.Encoding]::UTF8.GetString($secretBytes)
        $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::new($HttpMethod.ToUpperInvariant()), $fullUrl)
        $request.Headers.Add('x-api-key', $plainKey)
        $request.Headers.Add('Accept', 'application/json')

        if (-not [string]::IsNullOrWhiteSpace($JsonBody)) {
            $request.Content = [System.Net.Http.StringContent]::new($JsonBody, [System.Text.Encoding]::UTF8, 'application/json')
        }

        if (-not $Raw) {
            Write-Host ""
            Write-Host ">> $($HttpMethod.ToUpperInvariant()) $fullUrl" -ForegroundColor Cyan
        }

        $response = $client.SendAsync($request).GetAwaiter().GetResult()
        $statusCode = [int]$response.StatusCode
        $reason = $response.ReasonPhrase
        $rawBytes = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
        $respText = [System.Text.Encoding]::UTF8.GetString($rawBytes)

        if (-not $Raw) {
            $color = if ($statusCode -ge 200 -and $statusCode -lt 300) { 'Green' } else { 'Red' }
            Write-Host "<< HTTP $statusCode $reason" -ForegroundColor $color
        }

        if (-not [string]::IsNullOrWhiteSpace($respText)) {
            if ($Raw) {
                Write-Output $respText
            } else {
                try {
                    $parsed = $respText | ConvertFrom-Json -ErrorAction Stop
                    $pretty = $parsed | ConvertTo-Json -Depth 20
                    Write-Output $pretty
                }
                catch {
                    Write-Output $respText
                }
            }
        } elseif ($statusCode -ge 400) {
            throw "HTTP $statusCode $reason"
        }
    }
    finally {
        if ($null -ne $secretBytes) {
            [Array]::Clear($secretBytes, 0, $secretBytes.Length)
        }
        $plainKey = $null
        if ($null -ne $request) {
            $request.Headers.Clear()
            $request.Dispose()
        }
        if ($null -ne $response) { $response.Dispose() }
        if ($null -ne $client) { $client.Dispose() }
        if ($null -ne $handler) { $handler.Dispose() }
    }
}

function Start-InteractiveMenu {
    param(
        [string]$NormalizedBaseUrl,
        [string]$TargetName
    )

    while ($true) {
        Write-Host ""
        Write-Host "================ sub2api Admin 控制工具 ================" -ForegroundColor Cyan
        Write-Host "站点: $NormalizedBaseUrl"
        Write-Host "凭据: $TargetName"
        Write-Host "--------------------------------------------------------"
        Write-Host "  [1] 安全保存 / 更新控制密钥 (adm_sc_...)"
        Write-Host "  [2] 查看当前已保存密钥状态"
        Write-Host "  [3] 快捷验证：获取版本 / 分组列表 / 模型定价"
        Write-Host "  [4] 自定义调用 Admin API"
        Write-Host "  [5] 删除已保存的控制密钥"
        Write-Host "  [0] 退出"
        Write-Host "========================================================"

        $choice = (Read-Host "请选择操作 [0-5]").Trim()
        switch ($choice) {
            '1' {
                Save-ControlKeyInteractive -TargetName $TargetName -NormalizedBaseUrl $NormalizedBaseUrl | Out-Null
            }
            '2' {
                Show-SavedKeyStatus -TargetName $TargetName -NormalizedBaseUrl $NormalizedBaseUrl | Out-Null
            }
            '3' {
                Write-Host ""
                Write-Host "  [a] GET /api/v1/admin/system/version"
                Write-Host "  [b] GET /api/v1/admin/groups/all"
                Write-Host "  [c] GET /api/v1/admin/channels/model-pricing"
                Write-Host "  [d] GET /api/v1/admin/dashboard/stats"
                $sub = (Read-Host "选择预设接口 [a-d]").Trim().ToLowerInvariant()
                $presetPath = switch ($sub) {
                    'a' { '/api/v1/admin/system/version' }
                    'b' { '/api/v1/admin/groups/all' }
                    'c' { '/api/v1/admin/channels/model-pricing' }
                    'd' { '/api/v1/admin/dashboard/stats' }
                    default { '' }
                }
                if ($presetPath) {
                    try {
                        Invoke-Sub2ApiAdminRequest -NormalizedBaseUrl $NormalizedBaseUrl -TargetName $TargetName -HttpMethod 'GET' -ApiPath $presetPath -RequestTimeoutSec $TimeoutSec
                    } catch {
                        Write-Host "调用失败: $($_.Exception.Message)" -ForegroundColor Red
                    }
                }
            }
            '4' {
                $m = (Read-Host "HTTP 方法 [GET/POST/PUT/PATCH/DELETE，默认 GET]").Trim().ToUpperInvariant()
                if ([string]::IsNullOrWhiteSpace($m)) { $m = 'GET' }
                $p = Read-Host "API 路径（例如 /api/v1/admin/groups/12 或 groups/12）"
                $bodyInput = ''
                $fileInput = ''
                if ($m -in @('POST', 'PUT', 'PATCH')) {
                    $mode = (Read-Host "输入请求体方式：[1] 空请求体  [2] 直接输入 JSON  [3] 从 JSON 文件读取 [默认 1]").Trim()
                    if ($mode -eq '2') {
                        $bodyInput = Read-Host "JSON 请求体"
                    } elseif ($mode -eq '3') {
                        $fileInput = Read-Host "JSON 文件路径"
                    }
                }
                try {
                    Invoke-Sub2ApiAdminRequest -NormalizedBaseUrl $NormalizedBaseUrl -TargetName $TargetName -HttpMethod $m -ApiPath $p -JsonBody $bodyInput -JsonBodyFile $fileInput -RequestTimeoutSec $TimeoutSec
                } catch {
                    Write-Host "调用失败: $($_.Exception.Message)" -ForegroundColor Red
                }
            }
            '5' {
                $confirm = (Read-Host "确认删除凭据 '$TargetName'? (y/N)").Trim()
                if ($confirm -in @('y', 'Y', 'yes', 'YES')) {
                    Remove-SavedKey -TargetName $TargetName
                }
            }
            '0' {
                return
            }
            default {
                Write-Host "无效选项，请输入 0-5。" -ForegroundColor Yellow
            }
        }
    }
}

$normalizedBaseUrl = Resolve-NormalizedBaseUrl -Url $BaseUrl
$credTarget = Resolve-CredentialTarget -NormalizedBaseUrl $normalizedBaseUrl -ExplicitTarget $Target

switch ($Action) {
    'save' {
        Save-ControlKeyInteractive -TargetName $credTarget -NormalizedBaseUrl $normalizedBaseUrl | Out-Null
    }
    'status' {
        Show-SavedKeyStatus -TargetName $credTarget -NormalizedBaseUrl $normalizedBaseUrl | Out-Null
    }
    'delete' {
        Remove-SavedKey -TargetName $credTarget
    }
    'call' {
        Invoke-Sub2ApiAdminRequest -NormalizedBaseUrl $normalizedBaseUrl -TargetName $credTarget -HttpMethod $Method -ApiPath $Path -JsonBody $Body -JsonBodyFile $BodyFile -RequestTimeoutSec $TimeoutSec
    }
    'menu' {
        Start-InteractiveMenu -NormalizedBaseUrl $normalizedBaseUrl -TargetName $credTarget
    }
}
