// MFCApplication2Dlg.h: 헤더 파일
//


#pragma once

#include "afxwin.h" // MFC 기본 헤더 포함

// CMFCApplication2Dlg 대화 상자
class CMFCApplication2Dlg : public CDialogEx
{
public:
    // 생성자
    CMFCApplication2Dlg(CWnd* pParent = nullptr);

#ifdef AFX_DESIGN_TIME
    enum { IDD = IDD_MFCAPPLICATION2_DIALOG };
#endif

protected:
    virtual void DoDataExchange(CDataExchange* pDX); // DDX/DDV 지원


protected:
    HICON m_hIcon;

    // 메시지 맵 함수
    virtual BOOL OnInitDialog();
    afx_msg void OnPaint();
    afx_msg HCURSOR OnQueryDragIcon();
    DECLARE_MESSAGE_MAP()

private:
    CStringArray selectedFilePaths; // 여러 개의 파일 경로 저장


public:
    afx_msg void OnBnClickedButtonSelect();  // 파일 선택 버튼
    afx_msg void OnBnClickedButtonAnalyze(); // 분석 버튼
    afx_msg void OnBnClickedButtonReset();   // 초기화 버튼
    afx_msg void OnEnChangeStaticPath();
    afx_msg void OnBnClickedButtonCheck();
};