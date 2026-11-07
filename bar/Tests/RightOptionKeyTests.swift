import CoreGraphics
import XCTest

@testable import PhononBar

final class RightOptionKeyTests: XCTestCase {
    // NX_DEVICELALTKEYMASK / NX_DEVICERALTKEYMASK
    private let left: UInt64 = 0x20
    private let right: UInt64 = 0x40

    private func flags(_ device: UInt64, _ extra: CGEventFlags = []) -> CGEventFlags {
        CGEventFlags(rawValue: CGEventFlags.maskAlternate.rawValue | extra.rawValue | device)
    }

    func testRightOptionCounts() {
        XCTAssertTrue(RightOptionKey.isHeldAlone(flags(right)))
    }

    func testLeftOptionDoesNotCount() {
        XCTAssertFalse(RightOptionKey.isHeldAlone(flags(left)))
    }

    func testBothOptionsCount() {
        XCTAssertTrue(RightOptionKey.isHeldAlone(flags(left | right)))
    }

    func testRightOptionWithOtherModifiersDoesNotCount() {
        XCTAssertFalse(RightOptionKey.isHeldAlone(flags(right, .maskCommand)))
        XCTAssertFalse(RightOptionKey.isHeldAlone(flags(right, .maskControl)))
        XCTAssertFalse(RightOptionKey.isHeldAlone(flags(right, .maskShift)))
    }

    func testNoOptionDoesNotCount() {
        XCTAssertFalse(RightOptionKey.isHeldAlone(CGEventFlags(rawValue: 0)))
    }
}
