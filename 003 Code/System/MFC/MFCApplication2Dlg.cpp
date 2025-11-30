// MFCApplication2Dlg.cpp: 구현 파일
//

#include "pch.h"
#include "MFCApplication2.h"
#include "MFCApplication2Dlg.h"
#include "afxdialogex.h"
#include "ImageDialog.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstdio>


#ifdef _DEBUG
#define new DEBUG_NEW
#endif


void CMFCApplication2Dlg::DoDataExchange(CDataExchange* pDX)
{
    CDialogEx::DoDataExchange(pDX);
}

// Python 실행 함수
std::string runPythonScript(const std::vector<std::string>& imagePaths, CWnd* pWnd) {
    std::string command = "python C:\\working\\hbnu-swuniv-capstone-project-come-capstone25-CAPSTONE-InnoMedics\\003Code\\System\\analyze.py";

    for (const auto& path : imagePaths) {
        command += " " + path;
    }

    FILE* pipe = _popen(command.c_str(), "r");
    if (!pipe) return "Error";

    char buffer[256];
    std::string result;
    std::string lastLine;

    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        std::string line(buffer);

        // tqdm 출력에는 '\r'로 진행 상태가 갱신되므로, '\r' 처리
        size_t crPos = line.find('\r');
        if (crPos != std::string::npos)
            line = line.substr(crPos + 1);

        // 진행 상태를 다이얼로그에 출력
        CString progressText(line.c_str());
        pWnd->SetDlgItemText(IDC_STATIC_RESULT, progressText);

        // UI 즉시 갱신
        MSG msg;
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        result += line;
        lastLine = line;
    }

    _pclose(pipe);

    return lastLine;
}

// 파일 경로 저장 변수
CString selectedFilePath;

// 파일 선택 버튼
void CMFCApplication2Dlg::OnBnClickedButtonSelect()
{
    CFileDialog dlg(TRUE, _T("jpg"), NULL,
        OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_ALLOWMULTISELECT,
        _T("Image Files (*.jpg;*.png;*.bmp;*.jpeg;*.tif)|*.jpg;*.png;*.bmp;*.jpeg;*.tif|All Files (*.*)|*.*||"));

    dlg.m_ofn.nMaxFile = 1024; // 다중 선택을 위한 버퍼 크기 증가
    TCHAR szFile[1024] = { 0 };
    dlg.m_ofn.lpstrFile = szFile;

    if (dlg.DoModal() == IDOK) {
        selectedFilePaths.RemoveAll(); // 기존 경로 초기화

        POSITION pos = dlg.GetStartPosition();

        while (pos) {
            CString filePath = dlg.GetNextPathName(pos);
            selectedFilePaths.Add(filePath);
        }

        // 선택한 파일 경로를 UI에 표시 (한 줄씩)
        CString paths;
        for (int i = 0; i < selectedFilePaths.GetSize(); i++) {
            paths += selectedFilePaths[i] + _T("\r\n");
        }

        SetDlgItemText(IDC_STATIC_PATH, paths);
    }
}

// 분석 버튼
void CMFCApplication2Dlg::OnBnClickedButtonAnalyze()
{
    if (selectedFilePaths.GetSize() == 0) {
        MessageBox(_T("먼저 이미지를 선택하세요."), _T("오류"), MB_OK | MB_ICONERROR);
        return;
    }

    std::vector<std::string> imagePaths;
    for (int i = 0; i < selectedFilePaths.GetSize(); i++) {
        imagePaths.emplace_back(CT2A(selectedFilePaths[i])); // CString → std::string 변환
    }

    // tqdm 실시간 표시
    std::string result = runPythonScript(imagePaths, this);

    // 마지막 출력 파싱
    std::string resCnt, resPath;
    std::istringstream ss(result);
    std::getline(ss, resCnt, '/');
    std::getline(ss, resPath);

    CString cntText;
    cntText.Format(_T("객체 개수: %s"), CString(resCnt.c_str()));
    SetDlgItemText(IDC_STATIC_RESULT, cntText);

    // 결과 이미지 표시
    CString resPathCStr(resPath.c_str());
    ImageDialog imagedlg(resPathCStr);
    imagedlg.DoModal();
}

void CMFCApplication2Dlg::OnBnClickedButtonCheck()
{
    std::string checkPath = "C:\\Users\\yy\\Desktop\\test\\results\\prediction";
    
    CString cntText;
    cntText.Format(_T("이미지 주소: %s"), CString(checkPath.c_str()));
    SetDlgItemText(IDC_STATIC_RESULT, cntText);

    CString resPathCStr(checkPath.c_str());
    ImageDialog imagedlg(resPathCStr);
    imagedlg.DoModal();
}


// 초기화 버튼
void CMFCApplication2Dlg::OnBnClickedButtonReset()
{
    selectedFilePath = _T("");
    SetDlgItemText(IDC_STATIC_PATH, _T("파일을 선택하세요."));
    SetDlgItemText(IDC_STATIC_RESULT, _T("-"));
}

CMFCApplication2Dlg::CMFCApplication2Dlg(CWnd* pParent /*=nullptr*/)
    : CDialogEx(IDD_MFCAPPLICATION2_DIALOG, pParent)
{
    m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

BOOL CMFCApplication2Dlg::OnInitDialog()
{
    CDialogEx::OnInitDialog(); // 부모 클래스의 OnInitDialog 호출

    // 다이얼로그 초기화 작업 (예: 아이콘 설정)
    SetIcon(m_hIcon, TRUE);  // 큰 아이콘
    SetIcon(m_hIcon, FALSE); // 작은 아이콘

    // 여기서 초기화 작업 추가

    return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE, 설정하면 FALSE
}



BEGIN_MESSAGE_MAP(CMFCApplication2Dlg, CDialogEx)
    ON_BN_CLICKED(IDC_BUTTON_SELECT, &CMFCApplication2Dlg::OnBnClickedButtonSelect)
    ON_BN_CLICKED(IDC_BUTTON_ANALYZE, &CMFCApplication2Dlg::OnBnClickedButtonAnalyze)
    ON_BN_CLICKED(IDC_BUTTON_RESET, &CMFCApplication2Dlg::OnBnClickedButtonReset)
    ON_EN_CHANGE(IDC_STATIC_PATH, &CMFCApplication2Dlg::OnEnChangeStaticPath)
    ON_BN_CLICKED(IDC_BUTTON_CHECK, &CMFCApplication2Dlg::OnBnClickedButtonCheck)
END_MESSAGE_MAP()

void CMFCApplication2Dlg::OnEnChangeStaticPath()
{
    // TODO:  RICHEDIT 컨트롤인 경우, 이 컨트롤은
    // CDialogEx::OnInitDialog() 함수를 재지정 
    //하고 마스크에 OR 연산하여 설정된 ENM_CHANGE 플래그를 지정하여 CRichEditCtrl().SetEventMask()를 호출하지 않으면
    // ENM_CHANGE가 있으면 마스크에 ORed를 플래그합니다.

    // TODO:  여기에 컨트롤 알림 처리기 코드를 추가합니다.
}

