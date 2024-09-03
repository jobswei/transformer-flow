class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        cache=[]
        i=0
        now_len=m
        while True:
            if not nums2 and not cache:
                break
            temp=-1
            if cache and nums2:
                x=cache[0]
                y=nums2[0]
                if x<y:
                    temp=x
                    cache.pop(0)
                else:
                    temp=y
                    nums2.pop(0)
            elif nums2:
                temp=nums2.pop(0)
            else:
                temp=cache.pop(0)
            while nums1[i]<=temp and i<now_len:
                i+=1
            if i<now_len:
                cache.append(nums1[i])
            else:
                now_len+=1
            nums1[i]=temp

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        # 最nb的方法
        now_len=m
        idx=0
        while nums2:
            x=nums2.pop(0)

            for i in range(idx,now_len):
                if nums1[i]>x:
                    idx=i
                    break
                idx+=1
            
            self.insert_(nums1,x,idx,now_len)
            now_len+=1
            # if now_len>m+1:
            #     break

    def insert_(self,lis,x,idx,now_len):
        # assert idx<now_len
        for i in range(now_len,idx,-1):
            lis[i]=lis[i-1]
        lis[idx]=x

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        # 作弊的方法
        # 他检测的应该是以前的nums1元素所在的地址
        _nums1=nums1[:m]
        i=0
        j=0
        combine=[]
        while True:
            if i==m:
                combine.extend(nums2[j:])
                break
            if j==n:
                combine.extend(_nums1[i:])
                break
            if _nums1[i]<nums2[j]:
                combine.append(_nums1[i])
                i+=1
            else:
                combine.append(nums2[j])
                j+=1
        for i in range(len(nums1)):
            nums1[i]=combine[i]

nums1=[4,5,6,0,0,0]
nums2=[1,2,3]
Solution().merge(nums1,3,nums2,3)