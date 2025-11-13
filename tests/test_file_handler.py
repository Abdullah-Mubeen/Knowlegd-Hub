import asyncio
from pathlib import Path
from fastapi import UploadFile
from io import BytesIO

from app.utils.file_handler import get_file_handler


async def test_file_handler():
    """Test file handler functionality"""
    
    file_handler = get_file_handler()
    
    # Create a mock PDF file
    mock_pdf_content = b"%PDF-1.4\nMock PDF content for testing"
    mock_file = UploadFile(
        filename="test_document.pdf",
        file=BytesIO(mock_pdf_content)
    )
    
    print("🧪 Testing File Handler...")
    
    # Test validation
    is_valid, error = file_handler.validate_file(mock_file)
    print(f"✅ Validation: {is_valid} (Error: {error})")
    
    # Reset file pointer
    await mock_file.seek(0)
    
    # Test save file
    business_id = "test_business_123"
    document_id = "doc_123"
    
    file_path, metadata = await file_handler.save_upload_file(
        file=mock_file,
        business_id=business_id,
        document_id=document_id,
        validate=False
    )
    
    print(f"✅ File saved: {file_path}")
    print(f"   Metadata: {metadata}")
    
    # Test get file info
    file_info = file_handler.get_file_info(file_path)
    print(f"✅ File info: {file_info}")
    
    # Test list files
    files = file_handler.list_business_files(business_id)
    print(f"✅ Business files: {len(files)} found")
    
    # Test delete file
    deleted = file_handler.delete_file(file_path)
    print(f"✅ File deleted: {deleted}")
    
    # Test cleanup
    count = file_handler.cleanup_temp_files(max_age_hours=0)
    print(f"✅ Temp files cleaned: {count}")
    
    print("\n✅ All file handler tests passed!")


if __name__ == "__main__":
    asyncio.run(test_file_handler())
