using System;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Input;

namespace KTISV.Controls
{
    /// <summary>
    /// 數值輸入框:按 Enter 就送出。
    ///
    /// 這種欄位的綁定一定要用 <c>UpdateSourceTrigger=LostFocus</c>。若是逐字送出,
    /// 使用者想打「150」時,才打完第一個字就會被當成 1 Hz 夾成下限再寫回框裡,
    /// 第二個字根本打不下去。但只靠 LostFocus 又反直覺 —— 打完按 Enter 沒反應,
    /// 得再點一下別的地方才生效。所以在這裡補上 Enter,當場把文字寫回 ViewModel。
    /// </summary>
    public class NumericTextBox : TextBox
    {
        // 沿用 TextBox 的樣式與模板,否則佈景主題找不到這個型別,會變成沒有外觀的空框
        protected override Type StyleKeyOverride => typeof(TextBox);

        protected override void OnKeyDown(KeyEventArgs e)
        {
            if (e.Key is Key.Enter or Key.Return && !AcceptsReturn)
            {
                // 直接讓綁定送出,而不是把焦點趕走 —— 使用者常常要連改好幾個
                // 欄位,按一次 Enter 就被彈出輸入框是很煩的。
                BindingOperations.GetBindingExpressionBase(this, TextProperty)?.UpdateSource();
                e.Handled = true;
                return;
            }
            base.OnKeyDown(e);
        }
    }
}
