import numpy as np
import CalcModule  # 컴파일된 모듈 이름 (예: CalcModule)

def test_calculate_wake_induced_velocity():
    # 임의의 데이터 생성 (예시)
    num_wake = 100        # wake marker의 행 개수
    num_points = 40      # wake marker에 포함된 3D 포인트 개수
    # WakeMarker: 각 행에 num_points개의 3D 포인트가 이어진 배열 (즉, 열 개수는 num_points * 3)
    WakeMarker = np.random.rand(num_wake, num_points * 3).astype(np.float64)
    # Gamma_TR와 Gamma_Shed: (num_wake, num_points) 배열
    Gamma_TR = np.random.rand(num_wake, num_points).astype(np.float64)
    Gamma_Shed = np.random.rand(num_wake, num_points).astype(np.float64)
    # coloP: (P, 3) 형태의 콜로케이션 포인트 (여기서는 임의로 3개 생성)
    coloP = np.random.rand(3, 3).astype(np.float64)
    # rc: 각 패널에 해당하는 코어 반경, 길이는 num_points
    rc = np.linspace(0.1, 0.5, num_points).astype(np.float64)

    Vout = CalcModule.calculate_wake_induced_velocity(WakeMarker, Gamma_TR, Gamma_Shed, coloP, rc)
    print("calculate_wake_induced_velocity output:")
    print(Vout)

def test_calculate_wake_velocity():
    # 임의의 데이터 생성 (예시)
    num_wake = 100
    num_points = 40
    WakeMarker = np.random.rand(num_wake, num_points * 3).astype(np.float64)
    Gamma_TR = np.random.rand(num_wake, num_points).astype(np.float64)
    Gamma_Shed = np.random.rand(num_wake, num_points).astype(np.float64)
    # Gamma_BD: 블레이드 유도 와류 강도 (예시로 num_wake-1 행, 1열)
    Gamma_BD = np.random.rand(num_wake - 1, 1).astype(np.float64)
    # BDpoint, TEPoint: (num_wake, 3) 형태의 배열
    BDpoint = np.random.rand(num_wake, 3).astype(np.float64)
    TEPoint = np.random.rand(num_wake, 3).astype(np.float64)
    # rc: 패널용, rcBD: 블레이드용
    rc = np.linspace(0.1, 0.5, num_points).astype(np.float64)
    rcBD = np.linspace(0.2, 0.6, num_wake).astype(np.float64)
    '''
    print("WakeMarker : ", WakeMarker)
    print("Gamma_TR : ", Gamma_TR)
    print("Gamma_Shed : ", Gamma_Shed)
    print("BDpoint : ", BDpoint)
    print("TEPoint : ", TEPoint)
    print("rc : ", rc)
    print("rcBD : ", rcBD)
    '''
    Vout = CalcModule.calculate_wake_velocity(WakeMarker, Gamma_TR, Gamma_Shed, Gamma_BD,
                                                BDpoint, TEPoint, rc, rcBD)
    print("calculate_wake_velocity output:")
    print(Vout)

def test_Vortex_Vatistas():
    # Vortex_Vatistas 테스트
    A = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    B = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ColocationPoint = np.array([0.5, 0.5, 0.0], dtype=np.float64)
    vortexStrength = 1.0
    rc = 0.1
    n = 1.06
    result = CalcModule.Vortex_Vatistas(A, B, ColocationPoint, vortexStrength, rc, n)
    print("Vortex_Vatistas output:")
    print(result)

if __name__ == '__main__':
    print("Testing calculate_wake_induced_velocity...")
    test_calculate_wake_induced_velocity()
    print("\nTesting calculate_wake_velocity...")
    test_calculate_wake_velocity()
    print("\nTesting Vortex_Vatistas...")
    test_Vortex_Vatistas()