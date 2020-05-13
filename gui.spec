# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

SETUP_DIR = 'E:\\pytorch_software_cup\\pytorch_final\\'

a = Analysis(['gui.py',
	'vedio.py',
	'models.py',
	'utils\\augmentations.py',
	'utils\\datasets.py',
	'utils\\logger.py',
	'utils\\parse_config.py',
	'utils\\utils.py',
	'objecttracker\\KalmanFilterTracker.py'],
             pathex=['E:\\pytorch_software_cup\\pytorch_final'],
             binaries=[],
             datas=[(SETUP_DIR+'img','img'),(SETUP_DIR+'weights','weights'),(SETUP_DIR+'config','config'),(SETUP_DIR+'\\data\\samples','data\\samples'),(SETUP_DIR+'data\\vedio','data\\vedio'),],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='gui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='gui')
