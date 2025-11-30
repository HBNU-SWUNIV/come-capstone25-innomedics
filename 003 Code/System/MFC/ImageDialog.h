#pragma once
#include <vector>
#include <afxwin.h>
#include <afxcmn.h>
#include <gdiplus.h>

class ImageDialog : public CDialogEx
{
	DECLARE_DYNAMIC(ImageDialog)

public:
	ImageDialog(CString folderPath, CWnd* pParent = nullptr);
	virtual ~ImageDialog();

	enum { IDD = IDD_IMAGE_DIALOG };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);
	virtual BOOL OnInitDialog();

	DECLARE_MESSAGE_MAP()

private:
	HICON m_hIcon;
	CString m_folderPath;                       // prediction 폴더 경로
	std::vector<CString> m_resultFolders;       // 1_result, 2_result, ... 경로 목록

	std::vector<CImage> m_images;               // 4장의 이미지
	std::vector<Gdiplus::Image*> m_gdiplusImages; // GDI+ 이미지 포인터 벡터
	int m_currentResultIndex;                   // 현재 표시 중인 result 폴더 인덱스
	
	CStatic m_imgCtrl[4];                       // 4개의 static 이미지 컨트롤: histogram, mask, blurred
	CStatic m_staticName;
	CButton m_btnPrev;
	CButton m_btnNext;

	void LoadResultFolders();                   // result 폴더 목록 불러오기
	void ShowResultFolder(int index);           // 해당 폴더 이미지 표시
	void RepositionImageControls(int cx, int cy);

public:
	afx_msg void OnPaint();
	afx_msg void OnBnClickedButtonPrev();
	afx_msg void OnBnClickedButtonNext();
	afx_msg void OnSize(UINT nType, int cx, int cy);
};
